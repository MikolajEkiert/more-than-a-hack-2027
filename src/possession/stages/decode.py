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
) -> np.ndarray:
    """Most likely H / A / D sequence under switching costs, decoded per half (no smoothing lag).

    Each frame scores H, A and D from the stages' evidence (log-odds, divided by
    `frames_per_observation` because neighbouring frames repeat the same information). Changing
    state costs `turnover_cost` (H <-> A), `stop_cost` (play -> D) or `restart_cost` (D -> play),
    in nats, so a possession is created only when the evidence for it outweighs its two edges.
    Frames without team evidence cost H and A the same: the path keeps the last owner through
    loose balls and ball gaps, and never abstains inside a half. With `snap_start_s` or
    `snap_end_s` above zero the edges of the decoded stoppages are then moved onto the edges of
    the ball gap (`snap_stoppage_edges`).
    """
    eps = 1e-6
    alive = np.clip(np.asarray(p_alive, dtype=np.float64), eps, 1 - eps)
    home = np.asarray(p_home, dtype=np.float64)
    known = ~np.isnan(home)
    home = np.clip(np.where(known, home, 0.5), eps, 1 - eps)
    scale = 1.0 / frames_per_observation
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
    return labels
