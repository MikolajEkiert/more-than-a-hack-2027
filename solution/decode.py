"""Frame probabilities -> possessions. The step that decides spellF1.

Viterbi over three states H / A / D: maximise  sum_t log p_t(s_t)  -  penalties for every switch,
with one penalty for a team change (H <-> A) and one for entering or leaving a stoppage. A switch
must be paid for with accumulated evidence, so flicker disappears while a sharp, well-supported
change stays on its frame -- unlike a majority vote, which moves every boundary toward the middle
of its window.

Vectorised over a grid of penalty pairs so a whole sweep costs one pass over the match.
"""

import numpy as np

from solution import container as C

# Params
EPS = 1e-6


def viterbi_grid(prob, pairs):
    """prob (T, 3) for H, A, D; pairs [(team_penalty, dead_penalty), ...] -> codes (G, T)."""
    logp = np.log(np.clip(prob, EPS, 1.0))
    pairs = np.asarray(pairs, np.float64)
    G, T = len(pairs), len(prob)
    cost = np.zeros((G, 3, 3))
    cost[:, 0, 1] = cost[:, 1, 0] = pairs[:, 0]
    cost[:, :2, 2] = cost[:, 2, :2] = pairs[:, 1][:, None]
    back = np.empty((T, G, 3), np.int8)
    score = np.repeat(logp[:1], G, axis=0)
    for t in range(1, T):
        cand = score[:, :, None] - cost             # (G, from, to)
        back[t] = cand.argmax(1)
        score = cand.max(1) + logp[t]
    path = np.empty((G, T), np.int8)
    path[:, -1] = score.argmax(1)
    g = np.arange(G)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[t, g, path[:, t]]
    return path


def viterbi_tv(prob, team_cost, dead_cost):
    """Like viterbi_grid, but the H<->A switch penalty varies per frame: team_cost (G, T)."""
    logp = np.log(np.clip(prob, EPS, 1.0))
    G, T = team_cost.shape
    cost = np.zeros((G, 3, 3))
    cost[:, :2, 2] = cost[:, 2, :2] = np.asarray(dead_cost, np.float64)[:, None]
    back = np.empty((T, G, 3), np.int8)
    score = np.repeat(logp[:1], G, axis=0)
    for t in range(1, T):
        cost[:, 0, 1] = cost[:, 1, 0] = team_cost[:, t]
        cand = score[:, :, None] - cost
        back[t] = cand.argmax(1)
        score = cand.max(1) + logp[t]
    path = np.empty((G, T), np.int8)
    path[:, -1] = score.argmax(1)
    g = np.arange(G)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[t, g, path[:, t]]
    return path


def decode_tv(prob, half, team_cost, dead_cost):
    """Per-half decoding with frame-varying team penalties. team_cost (G, T_full)."""
    out = np.full((team_cost.shape[0], len(prob)), C.X, np.int8)
    for h in np.unique(half[half > 0]):
        sel = np.flatnonzero(half == h)
        out[:, sel] = viterbi_tv(prob[sel], team_cost[:, sel], dead_cost)
    return out


def min_dead(path, half, min_frames):
    """Remove dead-ball segments shorter than min_frames (a short D would split a possession).

    The gap takes the team on both sides if they agree, else the team before it.
    """
    if min_frames <= 0:
        return path
    out = path.copy()
    for h in np.unique(half[half > 0]):
        sel = np.flatnonzero(half == h)
        seg = out[sel]
        brk = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], brk, [len(seg)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if seg[a] != C.D or b - a >= min_frames or a == 0 or b == len(seg):
                continue
            before, after = seg[a - 1], seg[b]
            seg[a:b] = before if before in (C.H, C.A) else after
        out[sel] = seg
    return out


def snap(path, touch_h, touch_a, radius):
    """Move each direct H<->A hand-over to the nearest touch by the team taking over (+-radius)."""
    if radius <= 0:
        return path
    out = path.copy()
    T = len(out)
    idx = np.flatnonzero((out[1:] != out[:-1]) & np.isin(out[1:], (C.H, C.A))
                         & np.isin(out[:-1], (C.H, C.A))) + 1
    for t in idx:
        new, old = out[t], out[t - 1]
        touch = touch_h if new == C.H else touch_a
        a, b = max(t - radius, 1), min(t + radius + 1, T)
        hits = a + np.flatnonzero(np.nan_to_num(touch[a:b]) > 0)
        if not len(hits):
            continue
        t2 = int(hits[np.argmin(np.abs(hits - t))])
        if t2 < t and out[t2 - 1] == old:      # start the new possession earlier
            out[t2:t] = new
        elif t2 > t and np.all(out[t:t2] == new):  # start it later
            out[t:t2] = old
    return out


def decode(prob, half, pairs):
    """Decode each half separately; frames outside the halves are X. Returns (G, T) codes."""
    out = np.full((len(pairs), len(prob)), C.X, np.int8)
    for h in np.unique(half[half > 0]):
        sel = np.flatnonzero(half == h)
        out[:, sel] = viterbi_grid(prob[sel], pairs)
    return out
