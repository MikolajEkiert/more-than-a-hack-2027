"""Stage 3, decode: minimum-Bayes-risk decoding for spellF1 (task 14).

`decode.switch_cost` returns the single most likely H / A / D path. spellF1 does not reward the
most likely path: it counts possessions, each matched one-to-one to a true one by temporal IoU
(F1 averaged over IoU 0.3-0.7), and a D frame is never part of a possession. This decoder picks
the set of possessions with the largest expected spellF1 under the posterior of the same model:

1. The emissions and switching costs of `switch_cost` define a chain model over H / A / D,
   p(path) ~ exp(sharpness * (sum of emissions - sum of switching costs)). Its most likely path
   is exactly `switch_cost`'s for any `sharpness`; the posterior around it is not.
2. Forward filtering and backward sampling give `n_samples` paths per half from that posterior.
3. Every H / A run of a sampled path, and of the most likely path, is a candidate possession. Its
   match probability m is the share of sampled paths holding a run of the same team with IoU >= t
   to it, averaged over the metric's thresholds t.
4. F1 = 2 TP / (n_pred + n_true). Near its optimum F1*, adding a possession that is matched with
   probability m changes F1 by about (2 m - F1*) / (n_pred + n_true), so a possession is worth
   predicting when m exceeds `spell_cost`, about F1* / 2. The chosen set maximises
   sum(m - spell_cost) over candidates that do not overlap (weighted interval scheduling).
5. Chosen possessions get their team, every other graded frame D. A D run is never a possession,
   so frames the posterior cannot commit to separate possessions instead of joining them.
"""

import math

import numpy as np

from possession import features
from possession.match import Match
from possession.registry import register
from possession.stages.decode import snap_stoppage_edges

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)  # evaluation/possession_metrics.py SPELL_IOUS
MAX_OVERLAP = 12  # sampled runs inspected per candidate; beyond that every IoU is small
H, A, D = 0, 1, 2


def emission(p_alive, p_home, frames_per_observation, team_weight, alive_weight) -> np.ndarray:
    """(n, 3) log-scores of H, A, D per frame, exactly as `decode.switch_cost` builds them."""
    eps = 1e-6
    alive = np.clip(np.asarray(p_alive, dtype=np.float64), eps, 1 - eps)
    home = np.asarray(p_home, dtype=np.float64)
    known = ~np.isnan(home)
    home = np.clip(np.where(known, home, 0.5), eps, 1 - eps)
    scale = 1.0 / frames_per_observation
    live = alive_weight * scale * np.log(alive)
    return np.stack(
        [
            live + team_weight * scale * np.where(known, np.log(home), 0.0),
            live + team_weight * scale * np.where(known, np.log(1 - home), 0.0),
            alive_weight * scale * np.log(1 - alive),
        ],
        axis=1,
    )


def forward(scores: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """(T, 3) log forward messages of the chain with per-frame `scores` and switching `cost`.

    Plain Python over frames (three states, a sequential recurrence), like `decode._viterbi3`.
    """
    rows = scores.tolist()
    neg = (-np.asarray(cost, dtype=np.float64)).tolist()
    out = np.empty((len(rows), 3))
    prev = rows[0][:]
    out[0] = prev
    exp, log = math.exp, math.log
    for t in range(1, len(rows)):
        row = rows[t]
        cur = [0.0, 0.0, 0.0]
        for s in range(3):
            a0 = prev[0] + neg[0][s]
            a1 = prev[1] + neg[1][s]
            a2 = prev[2] + neg[2][s]
            top = a0 if a0 > a1 else a1
            top = top if top > a2 else a2
            cur[s] = top + log(exp(a0 - top) + exp(a1 - top) + exp(a2 - top)) + row[s]
        out[t] = cur
        prev = cur
    return out


def sample_paths(alpha: np.ndarray, cost: np.ndarray, n_samples: int, rng) -> np.ndarray:
    """(n_samples, T) int8 state paths drawn from the posterior by backward sampling."""
    steps = len(alpha)
    neg = -np.asarray(cost, dtype=np.float64)
    # cond[t, nxt, j]: P(state j at t | state nxt at t + 1), as cumulative sums over j
    logits = alpha[:-1, None, :] + neg.T[None, :, :]
    logits -= logits.max(axis=2, keepdims=True)
    cond = np.exp(logits)
    cond = np.cumsum(cond / cond.sum(axis=2, keepdims=True), axis=2)
    last = np.exp(alpha[-1] - alpha[-1].max())
    last = np.cumsum(last / last.sum())
    u = rng.random((steps, n_samples))
    paths = np.empty((n_samples, steps), dtype=np.int8)
    paths[:, -1] = np.minimum((u[-1][:, None] > last[None, :2]).sum(axis=1), 2)
    for t in range(steps - 2, -1, -1):
        c = cond[t][paths[:, t + 1]]
        paths[:, t] = (u[t][:, None] > c[:, :2]).sum(axis=1)
    return paths


def team_runs(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, stops, teams) of the H and A runs of one state path; stops exclusive."""
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(path)]))
    states = path[starts]
    keep = states != D
    return starts[keep], stops[keep], states[keep]


def match_probability(cands: np.ndarray, samples: list) -> np.ndarray:
    """Per candidate (start, stop, team): share of samples with a same-team run of IoU >= t,
    averaged over the metric's thresholds.

    A run counts only if the candidate covers more than half of it. The metric matches one to
    one, so two disjoint pieces of one long run cannot both be credited with it; for IoU >= 0.5
    the condition always holds, below 0.5 it gives the run to its larger piece only.
    """
    start, stop, team = cands[:, 0], cands[:, 1], cands[:, 2]
    hits = np.zeros(len(cands))
    for s_start, s_stop, s_team in samples:
        if not len(s_start):
            continue
        first = np.searchsorted(s_stop, start, side="right")
        end = np.searchsorted(s_start, stop, side="left")
        best = np.zeros(len(cands))
        last = len(s_start) - 1
        for j in range(MAX_OVERLAP):
            idx = first + j
            valid = idx < end
            if not valid.any():
                break
            k = np.minimum(idx, last)
            inter = np.minimum(stop, s_stop[k]) - np.maximum(start, s_start[k])
            union = np.maximum(stop, s_stop[k]) - np.minimum(start, s_start[k])
            same = valid & (s_team[k] == team) & (2 * inter > s_stop[k] - s_start[k])
            best = np.maximum(best, np.where(same, inter / np.maximum(union, 1), 0.0))
        hits += np.mean([best >= t for t in IOUS], axis=0)
    return hits / max(len(samples), 1)


def choose(cands: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Indices of the non-overlapping candidates with the largest total value (value > 0 only)."""
    keep = np.flatnonzero(value > 0)
    if not len(keep):
        return keep
    order = keep[np.lexsort((cands[keep, 0], cands[keep, 1]))]
    starts, stops, gains = cands[order, 0], cands[order, 1], value[order]
    prior = np.searchsorted(stops, starts, side="right") - 1  # last one ending at or before
    best = np.zeros(len(order) + 1)
    take = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        with_it = gains[i] + best[prior[i] + 1]
        if with_it > best[i]:
            best[i + 1] = with_it
            take[i] = True
        else:
            best[i + 1] = best[i]
    chosen = []
    i = len(order) - 1
    while i >= 0:
        if take[i]:
            chosen.append(order[i])
            i = prior[i]
        else:
            i -= 1
    return np.array(sorted(chosen), dtype=np.int64)


def decode_half(scores, cost, n_samples, spell_cost, rng) -> np.ndarray:
    """(T,) int8 states of one half: the chosen possessions, D elsewhere."""
    from possession.stages.decode import _viterbi3

    alpha = forward(scores, cost)
    paths = sample_paths(alpha, cost, n_samples, rng)
    samples = [team_runs(p) for p in paths]
    best = _viterbi3(scores, cost.tolist())
    pool = samples + [team_runs(best)]
    starts = np.concatenate([s[0] for s in pool])
    stops = np.concatenate([s[1] for s in pool])
    teams = np.concatenate([s[2] for s in pool]).astype(np.int64)
    cands = np.unique(np.stack([starts, stops, teams], axis=1), axis=0)
    value = match_probability(cands, samples) - spell_cost
    out = np.full(len(scores), D, dtype=np.int8)
    previous = None
    for i in choose(cands, value):
        a, b, k = (int(v) for v in cands[i])
        out[a:b] = k
        if previous is not None and previous[1] == a and previous[2] == k:
            out[a] = D  # two chosen runs of one team that touch would merge into one
        previous = (a, b, k)
    return out


@register("decode", "mbr")
def mbr(
    match: Match,
    p_alive,
    p_home,
    turnover_cost: float = 5.0,
    stop_cost: float = 12.0,
    restart_cost: float = 12.0,
    frames_per_observation: float = 5.0,
    team_weight: float = 1.0,
    alive_weight: float = 1.0,
    sharpness: float = 1.0,
    n_samples: int = 64,
    spell_cost: float = 0.3,
    seed: int = 0,
    snap_start_s: float = 0.0,
    snap_end_s: float = 0.0,
    restart_speed_m_s: float = 3.0,
) -> np.ndarray:
    """Possessions chosen for the largest expected spellF1 under the switching-cost posterior."""
    scores = emission(p_alive, p_home, frames_per_observation, team_weight, alive_weight)
    cost = np.array(
        [
            [0.0, turnover_cost, stop_cost],
            [turnover_cost, 0.0, stop_cost],
            [restart_cost, restart_cost, 0.0],
        ]
    )
    names = np.array(["H", "A", "D"])
    labels = np.full(match.n_frames, "X", dtype="<U1")
    for number in (1, 2):
        for start, stop in zip(*features.runs(match.half == number), strict=True):
            rng = np.random.default_rng([seed, number, int(start)])
            states = decode_half(
                sharpness * scores[start:stop], sharpness * cost, n_samples, spell_cost, rng
            )
            labels[start:stop] = names[states]
    if snap_start_s > 0 or snap_end_s > 0:
        labels = snap_stoppage_edges(match, labels, snap_start_s, snap_end_s, restart_speed_m_s)
    return labels
