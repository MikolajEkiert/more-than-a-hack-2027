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


def _viterbi3(emission: np.ndarray, cost: list[list[float]]) -> np.ndarray:
    """Best state path for (n, 3) log-emissions under a 3x3 matrix of switching costs (nats).

    Plain Python on purpose: three states and a sequential recurrence, where a per-frame numpy
    call would cost more than the arithmetic. Ties keep the lower state index, so it is
    deterministic.
    """
    n = len(emission)
    rows = emission.tolist()
    back = np.zeros((n, 3), dtype=np.int8)
    score = rows[0][:]
    for t in range(1, n):
        new = [0.0, 0.0, 0.0]
        pointers = [0, 0, 0]
        for state in range(3):
            best, arg = score[0] - cost[0][state], 0
            for prev in (1, 2):
                value = score[prev] - cost[prev][state]
                if value > best:
                    best, arg = value, prev
            new[state] = best + rows[t][state]
            pointers[state] = arg
        score = new
        back[t] = pointers
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


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
) -> np.ndarray:
    """Most likely H / A / D sequence under switching costs, decoded per half (no smoothing lag).

    Each frame scores H, A and D from the stages' evidence (log-odds, divided by
    `frames_per_observation` because neighbouring frames repeat the same information). Changing
    state costs `turnover_cost` (H <-> A), `stop_cost` (play -> D) or `restart_cost` (D -> play),
    in nats, so a possession is created only when the evidence for it outweighs its two edges.
    Frames without team evidence cost H and A the same: the path keeps the last owner through
    loose balls and ball gaps, and never abstains inside a half.
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
    return labels
