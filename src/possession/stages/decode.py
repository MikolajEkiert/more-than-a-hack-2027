"""Stage 3, decode: per-frame probabilities -> labels.

Contract: method(match, p_alive, p_home, **params) -> labels, <U1 (n,), each in {H, A, D, X}.
The pipeline overwrites ungraded frames with X afterwards, so decoders need not handle them.

This is where possession *structure* lives: spellF1 rewards whole possessions, so smoothing,
minimum durations and state models belong here. Remember the X rule (EVALUATION.md): the scorer
forward-fills X, so X is free inside a possession and expensive across a hand-over.
"""

import numpy as np

from possession import features
from possession.match import Match
from possession.registry import register
from starter.baseline import AWAY, HOME, NOBODY, smooth


def hard_codes(p_alive: np.ndarray, p_home: np.ndarray) -> np.ndarray:
    """int8 codes as starter/baseline.py uses them: HOME, AWAY, or NOBODY (dead / no evidence)."""
    codes = np.full(p_alive.shape[0], NOBODY, dtype=np.int8)
    alive = (p_alive >= 0.5) & ~np.isnan(p_home)
    codes[alive & (p_home >= 0.5)] = HOME
    codes[alive & (p_home < 0.5)] = AWAY
    return codes


@register("decode", "majority_vote")
def majority_vote(match: Match, p_alive, p_home, window: int = 50) -> np.ndarray:
    """Centred majority vote over +/- window frames (the reference baseline's smoother)."""
    return smooth(hard_codes(p_alive, p_home), window).astype("<U1")


TIE = 1e-9  # nats; scores closer than this are a tie


def _viterbi3(emission: np.ndarray, cost: list[list[float]]) -> np.ndarray:
    """Best state path for (n, 3) log-emissions under a 3x3 matrix of switching costs (nats).

    When staying and switching score the same, the path switches: a change of state then lands on
    the first frame that has evidence for the new state, not somewhere inside a stretch without
    evidence. That is the provider's rule (possession stays until the opponent touches the ball),
    and it treats both teams alike.

    Plain Python on purpose: three states and a sequential recurrence, where a per-frame numpy
    call would cost more than the arithmetic. Deterministic: equal switches keep the lower state.
    """
    n = len(emission)
    rows = emission.tolist()
    back = np.zeros((n, 3), dtype=np.int8)
    score = rows[0][:]
    for t in range(1, n):
        new = [0.0, 0.0, 0.0]
        pointers = [0, 0, 0]
        for state in range(3):
            best, arg = score[state], state
            for prev in range(3):
                if prev == state:
                    continue
                value = score[prev] - cost[prev][state]
                if value >= best - TIE and (arg == state or value > best + TIE):
                    arg = prev
                    best = max(best, value)
            new[state] = best + rows[t][state]
            pointers[state] = arg
        score = new
        back[t] = pointers
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_stoppage_edges(
    match: Match,
    labels: np.ndarray,
    start_s: float,
    end_s: float,
    restart_speed_m_s: float,
) -> np.ndarray:
    """Move the edges of decoded stoppages onto the edges of the ball gap they belong to.

    Players keep running for a moment after the whistle and start before the ball is seen again,
    so evidence from their speed turns late at the start of a stoppage and is vague at its end.
      * start: a stoppage decoded within `start_s` after the ball was lost, with no ball seen in
        between, began when the ball was lost
      * end: a restart decoded within `end_s` after the ball reappeared, with the ball seen all
        the way and moving at `restart_speed_m_s` when it reappeared, happened at the
        reappearance; a restart decoded up to `end_s` before the ball reappears is moved to it
    """
    out = labels.copy()
    seen = match.ball_seen
    fps = match.fps
    speed = features.ball_kinematics(match)["speed"]
    gap_start, gap_stop = features.runs(match.graded & ~seen)
    if not len(gap_start):
        return out
    dead_start, dead_stop = features.runs(labels == "D")
    n = len(labels)
    for start, stop in zip(dead_start.tolist(), dead_stop.tolist(), strict=True):
        if start_s > 0:
            i = int(np.searchsorted(gap_start, start, side="right")) - 1
            if (
                i >= 0
                and gap_start[i] < start <= min(gap_stop[i], gap_start[i] + start_s * fps)
                and match.half[gap_start[i]] == match.half[start]
                and (labels[gap_start[i] : start] != "D").all()
            ):
                out[gap_start[i] : start] = "D"
        if end_s <= 0 or stop >= n or labels[stop] not in ("H", "A"):
            continue
        k = int(np.searchsorted(gap_stop, stop, side="right")) - 1
        j = int(np.searchsorted(gap_stop, stop, side="left"))
        if k >= 0 and start < gap_stop[k] < stop and stop - gap_stop[k] <= end_s * fps:
            back = int(gap_stop[k])
            moving = np.nanmax(speed[back : min(back + fps // 2, n)], initial=0.0)
            if seen[back:stop].all() and moving >= restart_speed_m_s:
                out[back:stop] = labels[stop]
        elif j < len(gap_stop) and gap_start[j] <= stop < gap_stop[j] <= stop + end_s * fps:
            ahead = int(gap_stop[j])
            if (labels[stop:ahead] == labels[stop]).all() and match.half[ahead - 1] == match.half[
                stop
            ]:
                out[stop:ahead] = "D"
    return out


def _ball_zone(match: Match, margin_m: float) -> np.ndarray:
    """(n,) int8: +1 ball seen on or inside the lines, -1 seen beyond one by more than
    `margin_m`, 0 not seen or in between (the last zone seen holds over such frames)."""
    edge = features.ball_edge_distance(match)
    zone = np.zeros(match.n_frames, dtype=np.int8)
    zone[edge >= 0] = 1  # NaN compares False: a missing ball has no zone
    zone[edge < -margin_m] = -1
    return zone


def _crossings(zone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(exits, entries), bool (n,): the first frame seen outside after the ball was last seen
    inside, and the first frame seen inside after it was last seen outside."""
    exits = np.zeros(len(zone), dtype=bool)
    entries = np.zeros(len(zone), dtype=bool)
    known = np.flatnonzero(zone != 0)
    before, after = zone[known[:-1]], zone[known[1:]]
    exits[known[1:][(before == 1) & (after == -1)]] = True
    entries[known[1:][(before == -1) & (after == 1)]] = True
    return exits, entries


def snap_to_line_exit(
    match: Match, labels: np.ndarray, within_s: float, margin_m: float
) -> np.ndarray:
    """Begin a decoded stoppage when the tracked ball left the pitch shortly before it (Law 9).

    The ball is out of play once it has wholly crossed a boundary line, so a stoppage decoded
    within `within_s` after the ball was seen beyond a line by more than `margin_m`, with the
    ball not seen back inside in between and play decoded in between, began at the first frame
    the ball was seen outside.
    """
    out = labels.copy()
    zone = _ball_zone(match, margin_m)
    exits, _ = _crossings(zone)
    if not exits.any():
        return out
    index = np.arange(len(labels))
    last_exit = np.maximum.accumulate(np.where(exits, index, -1))
    last_seen = np.maximum.accumulate(np.where(zone != 0, index, -1))
    play = np.isin(labels, ("H", "A"))
    limit = within_s * match.fps
    starts, _ = features.runs(labels == "D")
    for start in starts.tolist():
        if start == 0 or not play[start - 1]:
            continue
        exit_ = int(last_exit[start - 1])
        if exit_ < 0 or zone[last_seen[start - 1]] != -1 or start - exit_ > limit:
            continue  # no exit, the ball was seen back inside since, or the exit is too early
        if match.half[exit_] == match.half[start] and play[exit_:start].all():
            out[exit_:start] = "D"
    return out


def snap_to_reentry(
    match: Match,
    labels: np.ndarray,
    within_s: float,
    margin_m: float,
    restart_speed_m_s: float,
) -> np.ndarray:
    """Move a decoded restart onto the moment the tracked ball came back onto the pitch.

    A throw-in restarts play when the ball enters the pitch from beyond the touchline. A restart
    decoded within `within_s` of the ball crossing from outside (beyond a line by more than
    `margin_m`) to inside, with the ball moving at `restart_speed_m_s` just after the crossing
    (so a ball carried or placed back in does not count), is moved to the crossing: earlier if
    the frames in between were decoded dead, later if they were decoded as the restarting team.
    The nearest crossing wins, the earlier one on a tie; each side keeps at least one frame.
    """
    out = labels.copy()
    _, entries = _crossings(_ball_zone(match, margin_m))
    entry = np.flatnonzero(entries)
    if not len(entry):
        return out
    n, fps = len(labels), match.fps
    speed = features.ball_kinematics(match)["speed"]
    limit = within_s * fps
    _, stops = features.runs(labels == "D")
    for restart in stops.tolist():
        if restart >= n or labels[restart] not in ("H", "A"):
            continue
        k = int(np.searchsorted(entry, restart))
        near = [int(c) for c in entry[max(k - 1, 0) : k + 1] if c != restart]
        near = [c for c in near if abs(c - restart) <= limit]
        if not near:
            continue
        cross = min(near, key=lambda c: (abs(c - restart), c))
        moving = np.nanmax(speed[cross : min(cross + fps // 2, n)], initial=0.0)
        if moving < restart_speed_m_s or match.half[cross] != match.half[restart]:
            continue
        team = labels[restart]
        if cross < restart and (labels[cross - 1 : restart] == "D").all():
            out[cross:restart] = team
        elif cross > restart and (labels[restart : cross + 1] == team).all():
            out[restart:cross] = "D"
    return out


SHORT_SPELL_POLICIES = ("keep", "dead", "merge")


def short_spells(
    match: Match,
    labels: np.ndarray,
    p_home,
    max_s: float,
    policy: str,
    max_llr: float = 0.0,
) -> np.ndarray:
    """Relabel short team spells sandwiched between two spells of the other team.

    A spell of one team shorter than `max_s` whose neighbours on both sides are spells of the
    other team becomes D ("dead": the stoppage keeps the neighbours apart without claiming a
    possession that may be noise) or the neighbours' team ("merge"); "keep" leaves it. With
    `max_llr` above zero only spells whose mean |team log-odds| (0 on frames without team
    evidence) is below it are touched. Spells are visited shortest first (then by start) and
    re-checked against the labels as they stand, so chains of short spells resolve one at a time.
    """
    if policy not in SHORT_SPELL_POLICIES:
        raise ValueError(
            f"unknown short_spell_policy {policy!r}; use one of {SHORT_SPELL_POLICIES}"
        )
    out = labels.copy()
    if policy == "keep" or max_s <= 0:
        return out
    eps = 1e-6
    home = np.asarray(p_home, dtype=np.float64)
    known = ~np.isnan(home)
    home = np.clip(np.where(known, home, 0.5), eps, 1 - eps)
    strength = np.where(known, np.abs(np.log(home) - np.log(1 - home)), 0.0)
    limit = max_s * match.fps
    spells = []
    for team in ("H", "A"):
        starts, stops = features.runs(labels == team)
        spells += [(b - a, a, b) for a, b in zip(starts.tolist(), stops.tolist(), strict=True)]
    n = len(labels)
    for length, start, stop in sorted(s for s in spells if s[0] < limit):
        team = out[start]
        other = "A" if team == "H" else "H"
        if start == 0 or stop >= n or out[start - 1] != other or out[stop] != other:
            continue
        if not (out[start:stop] == team).all():
            continue
        if max_llr > 0 and strength[start:stop].sum() / length >= max_llr:
            continue
        out[start:stop] = "D" if policy == "dead" else other
    return out


def shift_changes(labels: np.ndarray, frames: int) -> np.ndarray:
    """Move every direct H <-> A change `frames` earlier; the spell before keeps one frame."""
    out = labels.copy()
    if frames <= 0:
        return out
    play = np.isin(labels, ("H", "A"))
    edge = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    change = edge[play[edge] & play[edge - 1]]
    if not len(change):
        return out
    starts = np.concatenate(([0], edge))
    before = starts[np.searchsorted(starts, change) - 1]
    shift = np.minimum(int(frames), change - before - 1)
    for at, k in zip(change.tolist(), shift.tolist(), strict=True):
        if k > 0:
            out[at - k : at] = labels[at]
    return out


@register("decode", "switch_cost")
def switch_cost(
    match: Match,
    p_alive,
    p_home,
    turnover_cost: float = 3.0,
    stop_cost: float = 8.0,
    restart_cost: float = 8.0,
    frames_per_observation: float = 5.0,
    team_weight: float = 1.0,
    alive_weight: float = 1.0,
    snap_start_s: float = 0.0,
    snap_end_s: float = 0.0,
    restart_speed_m_s: float = 3.0,
    missing_frames_per_observation: float | None = None,
    line_snap_s: float = 0.0,
    line_margin_m: float = 0.5,
    reentry_snap_s: float = 0.0,
    short_spell_s: float = 0.0,
    short_spell_policy: str = "keep",
    short_spell_max_llr: float = 0.0,
    change_shift_frames: int = 0,
) -> np.ndarray:
    """Most likely H / A / D sequence under switching costs, decoded per half (no smoothing lag).

    Each frame scores H, A and D from the stages' evidence (log-odds, divided by
    `frames_per_observation` because neighbouring frames repeat the same information; by
    `missing_frames_per_observation` instead on frames without a tracked ball, if given, since
    evidence inside a ball gap hardly changes from frame to frame). Changing state costs
    `turnover_cost` (H <-> A), `stop_cost` (play -> D) or `restart_cost` (D -> play), in nats,
    so a possession is created only when the evidence for it outweighs its two edges. Frames
    without team evidence cost H and A the same: the path keeps the last owner through loose
    balls and ball gaps, and never abstains inside a half. Then, each step off at its default:
      * `snap_start_s`, `snap_end_s`: stoppage edges onto the edges of the ball gap
        (`snap_stoppage_edges`)
      * `line_snap_s`: a stoppage starts where the ball left the pitch (`snap_to_line_exit`)
      * `reentry_snap_s`: a restart where the ball came back in (`snap_to_reentry`)
      * `short_spell_s`, `short_spell_policy`, `short_spell_max_llr`: short spells between two
        spells of the other team become D or merge (`short_spells`)
      * `change_shift_frames`: every direct H <-> A change moves earlier (`shift_changes`)
    """
    if short_spell_policy not in SHORT_SPELL_POLICIES:
        raise ValueError(
            f"unknown short_spell_policy {short_spell_policy!r}; use one of {SHORT_SPELL_POLICIES}"
        )
    eps = 1e-6
    alive = np.clip(np.asarray(p_alive, dtype=np.float64), eps, 1 - eps)
    home = np.asarray(p_home, dtype=np.float64)
    known = ~np.isnan(home)
    home = np.clip(np.where(known, home, 0.5), eps, 1 - eps)
    if missing_frames_per_observation is None:
        scale = 1.0 / frames_per_observation
    else:
        scale = np.where(
            match.ball_seen, 1.0 / frames_per_observation, 1.0 / missing_frames_per_observation
        )
    live = alive_weight * scale * np.log(alive)
    emission = np.stack(
        [
            live + team_weight * scale * np.where(known, np.log(home), 0.0),
            live + team_weight * scale * np.where(known, np.log(1 - home), 0.0),
            alive_weight * scale * np.log(1 - alive),
        ],
        axis=1,
    )
    cost = [
        [0.0, turnover_cost, stop_cost],
        [turnover_cost, 0.0, stop_cost],
        [restart_cost, restart_cost, 0.0],
    ]
    names = np.array(["H", "A", "D"])
    labels = np.full(match.n_frames, "X", dtype="<U1")
    for number in (1, 2):
        for start, stop in zip(*features.runs(match.half == number), strict=True):
            labels[start:stop] = names[_viterbi3(emission[start:stop], cost)]
    if snap_start_s > 0 or snap_end_s > 0:
        labels = snap_stoppage_edges(match, labels, snap_start_s, snap_end_s, restart_speed_m_s)
    if line_snap_s > 0:
        labels = snap_to_line_exit(match, labels, line_snap_s, line_margin_m)
    if reentry_snap_s > 0:
        labels = snap_to_reentry(match, labels, reentry_snap_s, line_margin_m, restart_speed_m_s)
    if short_spell_s > 0 and short_spell_policy != "keep":
        labels = short_spells(
            match, labels, p_home, short_spell_s, short_spell_policy, short_spell_max_llr
        )
    if change_shift_frames > 0:
        labels = shift_changes(labels, change_shift_frames)
    return labels
