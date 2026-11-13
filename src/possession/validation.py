"""Validation on OTHER matches: the evidence for choosing between versions (RULES.md, task 02).

The public 5-minute window of the competition match may not be used to choose anything. Labelled
data from other games may. This module turns open matches into a `Match` plus per-frame H/A/D
labels, degrades them to resemble the automatic tracking we are given, and scores a pipeline on
them with the official metric code.

Source: Metrica Sports sample data, Sample Games 1-3
(https://github.com/metrica-sports/sample-data). Same provider as the competition tracking; game 3
is in the same EPTS format, games 1 and 2 are CSV. The files are downloaded on first use into
`.cache/external/` and are never committed or redistributed; credit Metrica Sports.

How the labels are derived (an algorithm, no hand labels):
  * D  where the corrected tracking has no ball position AND the events confirm a stoppage (a
       restart at the end of the stretch, a stoppage event at its start, or 3 s without a ball).
       Short stretches without either are tracking dropouts during play and stay live.
  * H/A elsewhere, under one of two readings of "possession" (`CONVENTIONS`):
      - "touch":      the team of the last on-ball event. The provider's event definition: a
                      player is in possession "as soon as they touch the ball".
      - "controlled": a touch counts only if the team then holds the ball for `MIN_HELD_FRAMES`
                      or completes a pass, shot or set piece; the possession is then dated from
                      that first touch. CHALLENGE.md's definition: a touch without control, a
                      deflection, or an aerial duel won without control is not a change.
    Which reading the competition ground truth follows is not known. Published figures point
    both ways (docs/DECISIONS.md), so every score is reported under both and a change must not
    lose under either.

What degradation cannot reproduce is stated in `degrade`. Treat scores here as a ranking signal
between versions, not as a forecast of the competition score.
"""

import csv
import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import possession_metrics as pm
from possession._paths import CACHE_DIR, CONFIGS_DIR
from possession.features import runs
from possession.match import AWAY, HOME, Match

_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"


def _csv_game(number: int) -> dict:
    stem = f"{_BASE}/Sample_Game_{number}/Sample_Game_{number}"
    return {
        "format": "csv",
        "home": f"{stem}_RawTrackingData_Home_Team.csv",
        "away": f"{stem}_RawTrackingData_Away_Team.csv",
        "events": f"{stem}_RawEventsData.csv",
    }


SOURCES = {
    "metrica-sg1": _csv_game(1),
    "metrica-sg2": _csv_game(2),
    "metrica-sg3": {
        "format": "epts",
        "tracking": f"{_BASE}/Sample_Game_3/Sample_Game_3_tracking.txt",
        "metadata": f"{_BASE}/Sample_Game_3/Sample_Game_3_metadata.xml",
        "events": f"{_BASE}/Sample_Game_3/Sample_Game_3_events.json",
    },
}
GAMES = tuple(sorted(SOURCES))
CONVENTIONS = ("touch", "controlled")
PITCH = (105.0, 68.0)
FPS = 25

# Events after which the acting team has the ball. BALL LOST, BALL OUT, CHALLENGE, FAULT RECEIVED
# and CARD never hand the ball over by themselves.
ON_BALL = ("SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT")
ACTS = ("SET PIECE", "PASS", "SHOT")  # on-ball events that show control on their own
LOSES = ("BALL LOST", "BALL OUT")
RESTART_TOKEN = "GOAL KICK"  # a goal kick is logged as a PASS (or a lost ball), not a SET PIECE
STOP_KINDS = ("BALL OUT", "FAULT RECEIVED")
STOP_TOKENS = ("GOAL", "OUT", "OFFSIDE", "HAND BALL", "END HALF", "FORCED", "FAULT")
MIN_HELD_FRAMES = 5  # 0.2 s on the ball: the shortest hold that counts as control
MIN_DEAD_FRAMES = 5  # shorter stretches without a ball are never stoppages
LONG_DEAD_S = 3.0  # a stretch without a ball this long is a stoppage even if no event says so
EVENT_SLACK = 5  # frames of tolerance when tying an event to an edge of a ball-less stretch
SPEED_SPAN = 2  # player speed = displacement over +/- this many frames


@dataclass(frozen=True)
class Event:
    """One provider event in a common form. Frames are 0-based video frames of that match."""

    start: int
    end: int
    team: int  # HOME or AWAY
    kind: str
    tokens: tuple[str, ...]  # subtype words, e.g. ("HEAD", "INTERCEPTION")
    period: int
    order: int


# -- reading the files ------------------------------------------------------------------------


def fetch(name: str = "metrica-sg3") -> dict[str, Path]:
    """Local paths of an external match's files, downloading the missing ones."""
    if name not in SOURCES:
        raise KeyError(f"unknown external match {name!r}; known: {list(GAMES)}")
    folder = CACHE_DIR / "external" / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, url in SOURCES[name].items():
        if kind == "format":
            continue
        path = folder / url.rsplit("/", 1)[1]
        if not path.is_file():
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URL)
                tmp.write_bytes(response.read())
            tmp.replace(path)
        paths[kind] = path
    return paths


def column_layout(root: ET.Element, player_ids: list[str]) -> list[tuple[int, int, list[int]]]:
    """[(first_frame_id, last_frame_id, slot of each player group)] from the EPTS metadata.

    A line carries only the players on the pitch, and which ones changes at every substitution;
    the metadata says which player each group belongs to in each frame range.
    """
    slot_of = {player: index for index, player in enumerate(player_ids)}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    layout = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        groups = [slot_of[channel_player[ref]] for ref in refs[::2]]  # x then y per player
        layout.append((int(spec.get("startFrame")), int(spec.get("endFrame")), groups))
    return sorted(layout)


def parse_tracking(path: Path, layout: list, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """EPTS sample tracking text -> (ball (n, 2), players (n, S, 2)), normalised 0-1, NaN missing.

    Line format: "<frame_id>:x,y;x,y;...:ball_x,ball_y" (no speed field, unlike the competition
    ATD). frame_id counts from 1; row i is frame_id i + 1. `layout` is `column_layout`'s result.
    """
    n = max(last for _, last, _ in layout)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, n_slots, 2), np.nan)
    spans = iter(layout)
    first, last, groups = next(spans)
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            while frame > last:
                first, last, groups = next(spans)
            flat = parts[1].replace(";", ",").split(",")
            if frame < first or len(flat) != len(groups) * 2:
                continue
            players[frame - 1, groups] = np.array(flat, dtype=np.float64).reshape(-1, 2)
            ball_fields = parts[2].split(",")
            if len(ball_fields) == 2:
                ball[frame - 1] = np.array(ball_fields, dtype=np.float64)
    return ball, players


def parse_tracking_csv(home: Path, away: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV sample tracking -> (ball (n, 2), players (n, S, 2), team_of_track (S,)); 0-1, NaN.

    Each file has three header lines, then "Period, Frame, Time, x, y per player ..., ball x, y".
    Home players take the first slots, away players the rest.
    """
    tables = [np.genfromtxt(path, delimiter=",", skip_header=3) for path in (home, away)]
    n = min(len(table) for table in tables)
    blocks = [table[:n, 3:-2].reshape(n, -1, 2) for table in tables]
    team_of_track = np.array(
        [HOME] * blocks[0].shape[1] + [AWAY] * blocks[1].shape[1], dtype=np.int8
    )
    return tables[0][:n, -2:], np.concatenate(blocks, axis=1), team_of_track


def events_from_json(path: Path, home_id: str) -> list[Event]:
    """Sample Game 3's event file (frames are 1-based there) as a sorted list of `Event`."""
    out = []
    for item in json.loads(Path(path).read_text())["data"]:
        subtypes = item.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        start = max(int(item["start"]["frame"]) - 1, 0)
        out.append(
            Event(
                start=start,
                end=max(int(item["end"]["frame"]) - 1, start),
                team=HOME if item["team"]["id"] == home_id else AWAY,
                kind=item["type"]["name"].upper(),
                tokens=tuple(s["name"].upper() for s in subtypes),
                period=int(item["period"]),
                order=int(item["index"]),
            )
        )
    return sorted(out, key=lambda e: (e.start, e.order))


def events_from_csv(path: Path) -> list[Event]:
    """Sample Games 1-2's event file (1-based frames, subtype words joined by '-')."""
    out = []
    with open(path, newline="") as handle:
        for order, row in enumerate(csv.DictReader(handle)):
            start = max(int(row["Start Frame"]) - 1, 0)
            out.append(
                Event(
                    start=start,
                    end=max(int(row["End Frame"]) - 1, start),
                    team=HOME if row["Team"].strip().lower() == "home" else AWAY,
                    kind=row["Type"].strip().upper(),
                    tokens=tuple(t for t in row["Subtype"].strip().upper().split("-") if t),
                    period=int(row["Period"]),
                    order=order,
                )
            )
    return sorted(out, key=lambda e: (e.start, e.order))


def speeds(xy: np.ndarray, fps: int) -> np.ndarray:
    """Speed (m/s) along axis 0 of an (n, ..., 2) position array; NaN where it cannot be formed."""
    out = np.full(xy.shape[:-1], np.nan)
    k = SPEED_SPAN
    step = xy[2 * k :] - xy[: -2 * k]
    out[k:-k] = np.hypot(step[..., 0], step[..., 1]) * fps / (2 * k)
    return out


# -- labels from events -----------------------------------------------------------------------


def _restarts(event: Event) -> bool:
    return event.kind == "SET PIECE" or RESTART_TOKEN in event.tokens


def _stops(event: Event) -> bool:
    return event.kind in STOP_KINDS or any(token in STOP_TOKENS for token in event.tokens)


def event_owner(events: list[Event], n_frames: int, convention: str = "touch") -> np.ndarray:
    """(n,) int8: the team in possession by the events, -1 before the first one.

    See the module docstring for the two conventions. Under "controlled" a team that only touches
    the ball (a header straight to an opponent, a save that goes out) never becomes the owner; a
    team that touches it and then holds, passes or shoots owns it from that first touch.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; known: {CONVENTIONS}")
    owner = np.full(n_frames, -1, dtype=np.int8)
    current = -1
    pending: dict[int, int | None] = {HOME: None, AWAY: None}
    for event in events:
        on_ball = event.kind in ON_BALL or _restarts(event)
        if not on_ball and event.kind not in LOSES:
            continue
        team = event.team
        if on_ball:
            pending[AWAY if team == HOME else HOME] = None  # the opponent's touch came to nothing
        if convention == "touch":
            if on_ball and team != current:
                owner[event.start :] = team
                current = team
            continue
        first_touch = pending[team]
        held = first_touch is not None and event.start - first_touch >= MIN_HELD_FRAMES
        acts = (
            event.kind in ACTS
            or _restarts(event)
            or (event.kind == "CARRY" and event.end - event.start >= MIN_HELD_FRAMES)
        )
        if acts or held:
            if team != current:
                owner[(first_touch if first_touch is not None else event.start) :] = team
                current = team
            pending[team] = None
        elif on_ball and first_touch is None:
            pending[team] = event.start
    return owner


def event_halves(events: list[Event], n_frames: int) -> np.ndarray:
    """(n,) int8: 1 or 2 inside a half, 0 outside.

    A half runs from its first event to its last; the second starts on the frame after the first
    ends, so the wait for its kick-off is graded, as it is in the competition match.
    """
    half = np.zeros(n_frames, dtype=np.int8)
    spans: dict[int, tuple[int, int]] = {}
    for event in events:
        first, last = spans.get(event.period, (n_frames, 0))
        spans[event.period] = (min(first, event.start), max(last, event.end))
    if 1 in spans:
        half[spans[1][0] : spans[1][1] + 1] = 1
    if 2 in spans:
        start = spans[1][1] + 1 if 1 in spans else spans[2][0]
        half[start : spans[2][1] + 1] = 2
    return half


def stoppage_mask(ball_seen: np.ndarray, events: list[Event], fps: int = FPS) -> np.ndarray:
    """(n,) bool: frames where play is stopped.

    A stretch without a ball position is a stoppage if it lasts `LONG_DEAD_S`, or a restart
    event falls at its end (or inside it), or a stoppage event falls at its start. Anything else
    without a ball is a dropout of the tracking during play.
    """
    n = len(ball_seen)
    restarts = np.array(sorted(e.start for e in events if _restarts(e)), dtype=np.int64)
    stops = np.array(sorted(e.end for e in events if _stops(e)), dtype=np.int64)
    dead = np.zeros(n, dtype=bool)
    for start, stop in zip(*runs(~ball_seen), strict=True):
        if stop - start < MIN_DEAD_FRAMES:
            continue
        long = stop - start >= LONG_DEAD_S * fps
        restarted = bool(
            ((restarts >= start + EVENT_SLACK) & (restarts <= stop + EVENT_SLACK)).any()
        )
        stopped = bool((np.abs(stops - (start - 1)) <= EVENT_SLACK).any())
        if long or restarted or stopped:
            dead[start:stop] = True
    return dead


def labels_from(owner: np.ndarray, half: np.ndarray, dead: np.ndarray) -> np.ndarray:
    """Per-frame H / A / D inside the halves, X outside (the format of a ground-truth file)."""
    labels = np.where(owner == HOME, "H", "A").astype("<U1")
    labels[dead | (owner < 0)] = "D"
    labels[half == 0] = "X"
    return labels


# -- matches ----------------------------------------------------------------------------------

_LOADED: dict[str, tuple[Match, list[Event]]] = {}


def _load(name: str) -> tuple[Match, list[Event]]:
    """(clean Match, events) of an external match, parsed once per process."""
    if name in _LOADED:
        return _LOADED[name]
    paths = fetch(name)
    if SOURCES[name]["format"] == "epts":
        root = ET.parse(paths["metadata"]).getroot()
        home_id = [team.get("id") for team in root.iter("Team")][0]
        roster = [(player.get("id"), player.get("teamId")) for player in root.iter("Player")]
        slots = [player for player, _ in roster]
        team_of_track = np.array([HOME if t == home_id else AWAY for _, t in roster], dtype=np.int8)
        layout = column_layout(root, slots)
        ball01, players01 = parse_tracking(paths["tracking"], layout, len(slots))
        events = events_from_json(paths["events"], home_id)
    else:
        ball01, players01, team_of_track = parse_tracking_csv(paths["home"], paths["away"])
        events = events_from_csv(paths["events"])

    scale = np.array(PITCH)
    ball_xy, players_xy = ball01 * scale, players01 * scale
    n = len(ball_xy)
    match = Match(
        game=name.upper(),
        fps=FPS,
        pitch=PITCH,
        ball=np.concatenate([ball_xy, np.full((n, 1), np.nan)], axis=1),
        players=np.concatenate([players_xy, speeds(players_xy, FPS)[..., None]], axis=2),
        team_of_track=team_of_track,
        half=event_halves(events, n),
        tagging=None,
    )
    match.validate()
    _LOADED[name] = (match, events)
    return _LOADED[name]


def load_external(name: str = "metrica-sg3", convention: str = "touch") -> tuple[Match, np.ndarray]:
    """(Match, ground-truth labels) of an external match, clean (as the provider published it)."""
    match, events = _load(name)
    owner = event_owner(events, match.n_frames, convention)
    return match, labels_from(owner, match.half, stoppage_mask(match.ball_seen, events))


# -- degradation ------------------------------------------------------------------------------

PROFILE_DIR = CONFIGS_DIR / "degradation"
DEGRADATION_KEYS = (
    "lost_ball_share",
    "lead_in_s",
    "free_gap_s",
    "visible_dead_prob",
    "visible_dead_s",
    "ball_offset_m",
    "ball_smooth_s",
    "players_visible",
)


def degradation_profiles() -> list[str]:
    """Names of the degradation profiles in configs/degradation/."""
    return sorted(path.stem for path in PROFILE_DIR.glob("*.json"))


def degradation_profile(name: str) -> dict:
    """The parameters of one named profile (see configs/degradation/README.md)."""
    path = PROFILE_DIR / f"{name}.json"
    if not path.is_file():
        raise KeyError(f"unknown degradation profile {name!r}; known: {degradation_profiles()}")
    params = json.loads(path.read_text())["params"]
    if set(params) != set(DEGRADATION_KEYS):
        unknown = sorted(set(params) - set(DEGRADATION_KEYS))
        missing = sorted(set(DEGRADATION_KEYS) - set(params))
        raise ValueError(f"{path.name}: unknown keys {unknown}, missing keys {missing}")
    return params


def degrade(
    match: Match, labels: np.ndarray, seed: int = 0, profile: str = "label_free", **overrides
) -> Match:
    """A copy of a clean external match made to look like automatic tracking.

    The amounts come from a named profile in configs/degradation/ (any of them can be overridden
    by keyword). What is done, mirroring `possession inspect gaps` on the competition tracking:
      * the ball is lost for whole stretches of 5 s or more, never for a moment. `lost_ball_share`
        of the live frames lose the ball: half of it just before stoppages (the tracker loses the
        ball, then play stops), half in stretches of `free_gap_s` anywhere in play
      * after some stoppages begin the ball stays visible for `visible_dead_s`, rolling on
      * corrected tracking pins the ball to the player in possession; a slowly varying offset of
        `ball_offset_m` (standard deviation) pulls it off again
      * corrected ball tracks change velocity within a frame at every event, an automatic one
        is smoother: a moving average of `ball_smooth_s` over the ball's position blurs kicks
      * tracks drop out until `players_visible` players are visible on average
    What it does not reproduce: identity switches, duplicated players, a ball visible but wrong,
    and which moments the real tracker loses (it is not random).
    """
    unknown = sorted(set(overrides) - set(DEGRADATION_KEYS))
    if unknown:
        raise TypeError(
            f"degrade() got unknown parameter(s) {unknown}; it takes {DEGRADATION_KEYS}"
        )
    p = {**degradation_profile(profile), **overrides}
    rng = np.random.default_rng(seed)
    n, fps = match.n_frames, match.fps
    ball = match.ball.copy()
    players = match.players.copy()
    alive = np.isin(labels, ("H", "A"))
    dead_starts, _ = runs(labels == "D")

    # 1. the ball stays visible, rolling on, at the start of some stoppages
    for start in dead_starts:
        if start == 0 or not alive[start - 1] or rng.random() >= p["visible_dead_prob"]:
            continue
        length = int(rng.uniform(*p["visible_dead_s"]) * fps)
        last = ball[start - 1, :2]
        back = max(start - 6, 0)
        velocity = (last - ball[back, :2]) / max(start - 1 - back, 1)
        if np.isnan(velocity).any() or np.isnan(last).any():
            continue
        stop = min(start + length, n)
        steps = np.arange(1, stop - start + 1)[:, None]
        ball[start:stop, :2] = last + velocity * steps

    # 2. a slowly varying offset between ball and possessor
    if p["ball_offset_m"] > 0:
        window = fps
        noise = rng.normal(size=(n + window, 2))
        smooth = np.stack(
            [np.convolve(noise[:, a], np.ones(window) / window, mode="valid")[:n] for a in (0, 1)],
            axis=1,
        )
        ball[:, :2] += smooth * p["ball_offset_m"] * np.sqrt(window)

    # 2b. the tracker's own smoothing of the ball, within each tracked stretch
    window = int(round(p["ball_smooth_s"] * fps))
    if window > 1:
        tracked = ~np.isnan(ball[:, 0])
        kernel = np.ones(window)
        count = np.convolve(tracked.astype(np.float64), kernel, mode="same")
        for axis in (0, 1):
            total = np.convolve(np.where(tracked, ball[:, axis], 0.0), kernel, mode="same")
            ball[:, axis] = np.where(tracked, total / np.maximum(count, 1.0), np.nan)

    # 3. the ball is lost for long stretches of live play
    hidden = np.zeros(n, dtype=bool)
    target = p["lost_ball_share"] * alive.sum()
    for index in rng.permutation(len(dead_starts)):
        if (hidden & alive).sum() >= target / 2:
            break
        start = dead_starts[index]
        hidden[max(start - int(rng.uniform(*p["lead_in_s"]) * fps), 0) : start] = True
    alive_frames = np.flatnonzero(alive)
    for _ in range(10 * len(dead_starts) + 100):
        if (hidden & alive).sum() >= target or not len(alive_frames):
            break
        start = int(rng.choice(alive_frames))
        hidden[start : start + int(rng.uniform(*p["free_gap_s"]) * fps)] = True
    ball[hidden] = np.nan

    # 4. tracks drop out
    visible = ~np.isnan(players[:, :, 0])
    mean_visible = visible[match.graded].sum(axis=1).mean() if match.graded.any() else 0.0
    if mean_visible > p["players_visible"]:
        drop = 1.0 - p["players_visible"] / mean_visible
        mean_out = 20.0 * fps
        mean_in = mean_out * (1.0 - drop) / drop
        for slot in range(players.shape[1]):
            frame = int(rng.exponential(mean_in))
            while frame < n:
                out = int(rng.exponential(mean_out)) + 1
                players[frame : frame + out, slot] = np.nan
                frame += out + int(rng.exponential(mean_in)) + 1

    return replace(match, ball=ball, players=players, _cache={})


# -- scoring ----------------------------------------------------------------------------------


def score(predicted: np.ndarray, truth: np.ndarray, bootstrap: int = 0) -> dict:
    """The official metrics of `predicted` against `truth` (both per-frame label arrays)."""
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted), bootstrap_n=bootstrap)
    out = {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "n_pred_possessions": int(result["n_pred_spells"]),
        "n_gt_possessions": int(result["n_gt_spells"]),
        "alive_dead_f1": round(result["alive_dead_f1"], 4),
        "team_acc_both_alive": round(result["team_acc_both_alive"], 4),
    }
    if bootstrap:
        out["spellF1_interval"] = [
            round(result["spell_bootstrap_low"], 4),
            round(result["spell_bootstrap_high"], 4),
        ]
    return out


def conditions(games=GAMES, seeds=(0, 1, 2), profile: str | None = "label_free"):
    """Yield (game, seed, match, {convention: truth}) for every validation condition.

    `profile=None` gives the clean tracking (one condition per game, seed None). The stoppages
    are the same under both conventions, so one degraded match serves both sets of labels.
    """
    for game in games:
        truths = {c: load_external(game, c)[1] for c in CONVENTIONS}
        clean = _load(game)[0]
        if profile is None:
            yield game, None, clean, truths
            continue
        for seed in seeds:
            yield game, seed, degrade(clean, truths["touch"], seed=seed, profile=profile), truths


def evaluate(run, games=GAMES, seeds=(0, 1, 2), profile: str | None = "label_free") -> list[dict]:
    """Score `run(match) -> labels` on every condition; one cell per game, seed and convention."""
    cells = []
    for game, seed, match, truths in conditions(games, seeds, profile):
        labels = run(match)
        for convention, truth in truths.items():
            cell = {"game": game, "seed": seed, "convention": convention}
            cells.append({**cell, **score(labels, truth)})
    return cells


def summarise(cells: list[dict], key: str = "spellF1") -> dict:
    """Means of `key` over the cells: overall, per convention, per game, per game and convention."""

    def mean(rows):
        return round(float(np.mean([row[key] for row in rows])), 4) if rows else None

    def pick(game=None, convention=None):
        return [
            c
            for c in cells
            if (game is None or c["game"] == game)
            and (convention is None or c["convention"] == convention)
        ]

    games = sorted({c["game"] for c in cells})
    return {
        "mean": mean(cells),
        "by_convention": {k: mean(pick(convention=k)) for k in CONVENTIONS},
        "by_game": {g: mean(pick(game=g)) for g in games},
        "by_game_convention": {g: {k: mean(pick(g, k)) for k in CONVENTIONS} for g in games},
    }


MIN_GAIN = 0.02  # mean spellF1 a change must add
MIN_GAMES_WON = 2 / 3  # share of the external matches it must win on


def compare(candidate: list[dict], reference: list[dict]) -> dict:
    """Paired comparison of two `evaluate` results under the acceptance rule fixed in advance.

    A change is accepted if its mean spellF1 gain is at least `MIN_GAIN`, it does not lose under
    either label convention, and it wins on at least `MIN_GAMES_WON` of the matches.
    """
    a, b = summarise(candidate), summarise(reference)
    gain = round(a["mean"] - b["mean"], 4)
    by_convention = {
        k: round(a["by_convention"][k] - b["by_convention"][k], 4) for k in CONVENTIONS
    }
    by_game = {g: round(a["by_game"][g] - b["by_game"][g], 4) for g in a["by_game"]}
    won = sum(value > 0 for value in by_game.values())
    accepted = (
        gain >= MIN_GAIN
        and all(value >= 0 for value in by_convention.values())
        and won >= MIN_GAMES_WON * len(by_game) - 1e-9
    )
    return {
        "gain": gain,
        "gain_by_convention": by_convention,
        "gain_by_game": by_game,
        "games_won": won,
        "n_games": len(by_game),
        "accepted": bool(accepted),
    }


# -- calibration against the published baselines ----------------------------------------------

# EVALUATION.md, "See the disagreement for yourself": the four organiser baselines scored on the
# full match. window (frames) -> (spellF1, macroF1). These are published documentation, not the
# ground truth; they are the only full-match numbers anyone outside the organisers has.
PUBLISHED_FULL_MATCH = {
    0: (0.0486, 0.6532),
    25: (0.1996, 0.6752),
    50: (0.2433, 0.6724),
    125: (0.2242, 0.6356),
}


def baseline_family(match: Match) -> dict[int, np.ndarray]:
    """The organiser baseline's labels on `match` for each published smoothing window."""
    from possession import registry, stages  # noqa: F401  (importing stages registers the methods)

    p_alive = registry.get("ball_state", "ball_visible").fn(match)
    p_home = registry.get("team", "nearest_player").fn(match)
    vote = registry.get("decode", "majority_vote").fn
    out = {}
    for window in PUBLISHED_FULL_MATCH:
        labels = np.asarray(vote(match, p_alive, p_home, window=window)).astype("<U1")
        labels[~match.graded] = "X"
        out[window] = labels
    return out


def baseline_residual(games=GAMES, seeds=(0, 1), profile: str = "label_free", **overrides) -> dict:
    """How far the baseline family on the degraded external matches is from the published scores.

    Returns, per convention, the root mean squared difference over the eight published numbers
    and the family's mean scores, plus label-free aggregates of the degraded tracking.
    """
    scores = {c: {w: [] for w in PUBLISHED_FULL_MATCH} for c in CONVENTIONS}
    tracked, visible = [], []
    for game in games:
        truths = {c: load_external(game, c)[1] for c in CONVENTIONS}
        clean = _load(game)[0]
        for seed in seeds:
            match = degrade(clean, truths["touch"], seed=seed, profile=profile, **overrides)
            tracked.append(float(match.ball_seen[match.graded].mean()))
            seen = ~np.isnan(match.players[match.graded, :, 0])
            visible.append(float(seen.sum(axis=1).mean()))
            for window, labels in baseline_family(match).items():
                for convention, truth in truths.items():
                    result = score(labels, truth)
                    scores[convention][window].append((result["spellF1"], result["macroF1"]))
    out = {
        "ball_tracked_share": round(float(np.mean(tracked)), 3),
        "players_visible": round(float(np.mean(visible)), 2),
    }
    for convention in CONVENTIONS:
        means = {w: np.mean(scores[convention][w], axis=0) for w in PUBLISHED_FULL_MATCH}
        errors = [means[w] - np.array(PUBLISHED_FULL_MATCH[w]) for w in PUBLISHED_FULL_MATCH]
        out[convention] = {
            "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 4),
            "scores": {str(w): [round(float(v), 4) for v in means[w]] for w in means},
        }
    out["rmse"] = round(float(np.mean([out[c]["rmse"] for c in CONVENTIONS])), 4)
    return out


CALIBRATION_GRID = {
    "lost_ball_share": (0.08, 0.15, 0.25, 0.35),
    "ball_offset_m": (0.5, 1.0, 1.5),
    "visible_dead": ((0.5, (0.5, 3.0)), (0.9, (1.0, 6.0))),  # (probability, seconds)
}


def calibrate(games=GAMES, seeds=(0, 1), base: str = "label_free", progress=None) -> list[dict]:
    """Search `CALIBRATION_GRID` around the `base` profile; rows sorted best first.

    The loss is the mean over both label conventions of the RMSE to `PUBLISHED_FULL_MATCH`.
    """
    rows = []
    for share in CALIBRATION_GRID["lost_ball_share"]:
        for offset in CALIBRATION_GRID["ball_offset_m"]:
            for probability, seconds in CALIBRATION_GRID["visible_dead"]:
                overrides = {
                    "lost_ball_share": share,
                    "ball_offset_m": offset,
                    "visible_dead_prob": probability,
                    "visible_dead_s": list(seconds),
                }
                result = baseline_residual(games, seeds, profile=base, **overrides)
                rows.append({"overrides": overrides, **result})
                if progress:
                    progress(rows[-1])
    return sorted(rows, key=lambda row: row["rmse"])


def write_profile(name: str, params: dict, source: str, evidence: dict | None = None) -> Path:
    """Write configs/degradation/<name>.json (the file both documents and defines a profile)."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{name}.json"
    body = {"name": name, "source": source, "params": params}
    if evidence is not None:
        body["evidence"] = evidence
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path
