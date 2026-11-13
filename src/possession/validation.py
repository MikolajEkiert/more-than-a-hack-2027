"""Validation on OTHER matches: the evidence for choosing between versions (RULES.md, task 02).

The public 5-minute window of the competition match may not be used to choose anything. Labelled
data from other games may. This module turns an open match into a `Match` plus per-frame H/A/D
labels, degrades it to resemble the automatic tracking we are given, and scores a pipeline on it
with the official metric code.

Source: Metrica Sports sample data, Sample Game 3 (https://github.com/metrica-sports/sample-data).
Same provider and the same EPTS format as the competition tracking. The files are downloaded on
first use into `.cache/external/` and are never committed or redistributed; credit Metrica Sports.

How the labels are derived (an algorithm, no hand labels):
  * D  where the corrected tracking has no ball position. In this dataset every restart event
       coincides with the end of such a run and every stoppage event with the start of one.
  * H/A elsewhere: the team of the last on-ball event (set piece, pass, carry, recovery, shot).
       A lost ball stays with the team that lost it until the opponent's next on-ball event,
       which is the provider's own definition of possession.

What degradation cannot reproduce is stated in `degrade`. Treat scores here as a ranking signal
between versions, not as a forecast of the competition score.
"""

import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

import possession_metrics as pm
from possession._paths import CACHE_DIR
from possession.features import runs
from possession.match import AWAY, HOME, Match

_BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
SOURCES = {
    "metrica-sg3": {
        "tracking": f"{_BASE}/Sample_Game_3/Sample_Game_3_tracking.txt",
        "metadata": f"{_BASE}/Sample_Game_3/Sample_Game_3_metadata.xml",
        "events": f"{_BASE}/Sample_Game_3/Sample_Game_3_events.json",
    }
}
# Events after which the acting team owns the ball. BALL LOST, BALL OUT, CHALLENGE, FAULT RECEIVED
# and CARD never hand the ball over by themselves.
ON_BALL = ("SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT")
MIN_DEAD_FRAMES = 5  # shorter runs without a ball are tracking dropouts, not stoppages
SPEED_SPAN = 2  # player speed = displacement over +/- this many frames


def fetch(name: str = "metrica-sg3") -> dict[str, Path]:
    """Local paths of an external match's files, downloading the missing ones."""
    if name not in SOURCES:
        raise KeyError(f"unknown external match {name!r}; known: {sorted(SOURCES)}")
    folder = CACHE_DIR / "external" / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, url in SOURCES[name].items():
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
    """Sample-data tracking text -> (ball (n, 2), players (n, S, 2)), normalised 0-1, NaN missing.

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


def speeds(xy: np.ndarray, fps: int) -> np.ndarray:
    """Speed (m/s) along axis 0 of an (n, ..., 2) position array; NaN where it cannot be formed."""
    out = np.full(xy.shape[:-1], np.nan)
    k = SPEED_SPAN
    step = xy[2 * k :] - xy[: -2 * k]
    out[k:-k] = np.hypot(step[..., 0], step[..., 1]) * fps / (2 * k)
    return out


def event_labels(events: list[dict], n_frames: int, home_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(owner (n,) int8: HOME, AWAY or -1 before the first on-ball event, half (n,) int8).

    Frames are 0-based (event frame - 1). A half runs from its first event to its last; the second
    starts on the frame after the first ends, so the wait for its kick-off is graded, as it is in
    the competition match.
    """
    owner = np.full(n_frames, -1, dtype=np.int8)
    on_ball = sorted(
        (e for e in events if e["type"]["name"] in ON_BALL),
        key=lambda e: (e["start"]["frame"], e["index"]),
    )
    for event in on_ball:
        team = HOME if event["team"]["id"] == home_id else AWAY
        owner[max(event["start"]["frame"] - 1, 0) :] = team

    half = np.zeros(n_frames, dtype=np.int8)
    spans = {}
    for event in events:
        first, last = spans.get(event["period"], (n_frames, 0))
        spans[event["period"]] = (
            min(first, event["start"]["frame"] - 1),
            max(last, event["end"]["frame"] - 1),
        )
    if 1 in spans:
        half[spans[1][0] : spans[1][1] + 1] = 1
    if 2 in spans:
        start = spans[1][1] + 1 if 1 in spans else spans[2][0]
        half[start : spans[2][1] + 1] = 2
    return owner, half


def labels_from(ball_seen: np.ndarray, owner: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Per-frame H / A / D inside the halves, X outside (the format of a ground-truth file)."""
    dead = ~ball_seen
    for start, stop in zip(*runs(dead), strict=True):
        if stop - start < MIN_DEAD_FRAMES:
            dead[start:stop] = False
    labels = np.where(owner == HOME, "H", "A").astype("<U1")
    labels[dead | (owner < 0)] = "D"
    labels[half == 0] = "X"
    return labels


def load_external(name: str = "metrica-sg3") -> tuple[Match, np.ndarray]:
    """(Match, ground-truth labels) of an external match, clean (as the provider published it)."""
    paths = fetch(name)
    root = ET.parse(paths["metadata"]).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    roster = [(player.get("id"), player.get("teamId")) for player in root.iter("Player")]
    slots = [player for player, _ in roster]
    team_of_track = np.array([HOME if t == home_id else AWAY for _, t in roster], dtype=np.int8)
    pitch = (105.0, 68.0)
    fps = 25

    ball01, players01 = parse_tracking(paths["tracking"], column_layout(root, slots), len(slots))
    scale = np.array(pitch)
    ball_xy, players_xy = ball01 * scale, players01 * scale
    n = len(ball_xy)
    ball = np.concatenate([ball_xy, np.full((n, 1), np.nan)], axis=1)
    players = np.concatenate([players_xy, speeds(players_xy, fps)[..., None]], axis=2)

    events = json.loads(paths["events"].read_text())["data"]
    owner, half = event_labels(events, n, home_id)
    match = Match(
        game=name.upper(),
        fps=fps,
        pitch=pitch,
        ball=ball,
        players=players,
        team_of_track=team_of_track,
        half=half,
        tagging=None,
    )
    match.validate()
    return match, labels_from(match.ball_seen, owner, half)


def degrade(
    match: Match,
    labels: np.ndarray,
    seed: int = 0,
    lost_ball_share: float = 0.15,
    lead_in_s: tuple[float, float] = (2.0, 8.0),
    free_gap_s: tuple[float, float] = (5.0, 20.0),
    visible_dead_prob: float = 0.5,
    visible_dead_s: tuple[float, float] = (0.5, 3.0),
    ball_offset_m: float = 0.5,
    players_visible: float = 20.5,
) -> Match:
    """A copy of a clean external match made to look like automatic tracking.

    Mirrors what `possession inspect gaps` shows on the competition tracking (aggregates only):
      * the ball is lost for whole stretches of 5 s or more, never for a moment. `lost_ball_share`
        of the live frames lose the ball: half of it just before stoppages (the tracker loses the
        ball, then play stops), half in stretches of `free_gap_s` anywhere in play
      * after some stoppages begin the ball stays visible for `visible_dead_s`, rolling on
      * corrected tracking pins the ball to the player in possession; a slowly varying offset of
        `ball_offset_m` (standard deviation) pulls it off again
      * tracks drop out until `players_visible` players are visible on average
    What it does not reproduce: identity switches, duplicated players, a ball visible but wrong,
    and which moments the real tracker loses (it is not random). The shares above are parameters,
    not measurements: the true split of our gaps into stoppage and play is unknown without labels.
    """
    rng = np.random.default_rng(seed)
    n, fps = match.n_frames, match.fps
    ball = match.ball.copy()
    players = match.players.copy()
    alive = np.isin(labels, ("H", "A"))
    dead_starts, dead_stops = runs(labels == "D")

    # 1. the ball stays visible, rolling on, at the start of some stoppages
    for start in dead_starts:
        if start == 0 or not alive[start - 1] or rng.random() >= visible_dead_prob:
            continue
        length = int(rng.uniform(*visible_dead_s) * fps)
        last = ball[start - 1, :2]
        back = max(start - 6, 0)
        velocity = (last - ball[back, :2]) / max(start - 1 - back, 1)
        if np.isnan(velocity).any() or np.isnan(last).any():
            continue
        stop = min(start + length, n)
        steps = np.arange(1, stop - start + 1)[:, None]
        ball[start:stop, :2] = last + velocity * steps

    # 2. a slowly varying offset between ball and possessor
    if ball_offset_m > 0:
        window = fps
        noise = rng.normal(size=(n + window, 2))
        smooth = np.stack(
            [np.convolve(noise[:, a], np.ones(window) / window, mode="valid")[:n] for a in (0, 1)],
            axis=1,
        )
        ball[:, :2] += smooth * ball_offset_m * np.sqrt(window)

    # 3. the ball is lost for long stretches of live play
    hidden = np.zeros(n, dtype=bool)
    target = lost_ball_share * alive.sum()
    order = rng.permutation(len(dead_starts))
    for index in order:
        if (hidden & alive).sum() >= target / 2:
            break
        start = dead_starts[index]
        hidden[max(start - int(rng.uniform(*lead_in_s) * fps), 0) : start] = True
    alive_frames = np.flatnonzero(alive)
    for _ in range(10 * len(dead_starts) + 100):
        if (hidden & alive).sum() >= target or not len(alive_frames):
            break
        start = int(rng.choice(alive_frames))
        hidden[start : start + int(rng.uniform(*free_gap_s) * fps)] = True
    ball[hidden] = np.nan

    # 4. tracks drop out
    visible = ~np.isnan(players[:, :, 0])
    mean_visible = visible[match.graded].sum(axis=1).mean() if match.graded.any() else 0.0
    if mean_visible > players_visible:
        drop = 1.0 - players_visible / mean_visible
        mean_out = 20.0 * fps
        mean_in = mean_out * (1.0 - drop) / drop
        for slot in range(players.shape[1]):
            frame = int(rng.exponential(mean_in))
            while frame < n:
                out = int(rng.exponential(mean_out)) + 1
                players[frame : frame + out, slot] = np.nan
                frame += out + int(rng.exponential(mean_in)) + 1

    return replace(match, ball=ball, players=players, _cache={})


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
