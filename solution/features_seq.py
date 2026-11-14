"""Sequence features for LightGBM: patterns of events over time, not just rolling means.

v1 already has rolling means of the key series and last/next-touch team. What it cannot express
is the order and rhythm of play, which is what CHALLENGE.md uses to define possession ("a team
completes more than two consecutive passes", "a player keeps the ball for a few seconds"). These
blocks add it as plain columns. Separate module on purpose: v1's cached features keep their keys.

    feature_fn = features.with_blocks(*features.V1_BLOCKS, *features_seq.SEQ_BLOCKS)
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from solution import features as F

# Params
CAP = 750                    # frames (30 s) cap for since / until clocks
K_EVENTS = 3                 # last / next touches described
COUNT_WIN = (25, 125, 250)   # touch counts over 1 s, 5 s, 10 s
RUN_CAP = 6


def _touch_events(f):
    """Collapse per-frame touch flags into events: (frame of first contact, team +1/-1)."""
    th = np.nan_to_num(f["touch_h"]) > 0
    ta = np.nan_to_num(f["touch_a"]) > 0
    sign = np.where(th, 1, np.where(ta, -1, 0))
    start = np.flatnonzero((sign != 0) & np.concatenate(([True], sign[1:] != sign[:-1])))
    return start, sign[start]


def touch_sequence_block(ctx, f):
    T = ctx["T"]
    ev_t, ev_s = _touch_events(f)
    idx = np.arange(T)
    new = {}
    n = len(ev_t)
    # j = index of the last event at or before t (-1 if none)
    j = np.searchsorted(ev_t, idx, side="right") - 1
    for k in range(K_EVENTS):
        jj = j - k
        ok = jj >= 0
        new["last{}_team".format(k + 1)] = np.where(ok, ev_s[np.maximum(jj, 0)] if n else 0, 0).astype(np.float64)
        new["last{}_ago".format(k + 1)] = np.where(ok, np.minimum(idx - ev_t[np.maximum(jj, 0)] if n else CAP, CAP), CAP).astype(np.float64)
        nn = j + 1 + k
        ok = nn < n
        new["next{}_team".format(k + 1)] = np.where(ok, ev_s[np.minimum(nn, max(n - 1, 0))] if n else 0, 0).astype(np.float64)
        new["next{}_in".format(k + 1)] = np.where(ok, np.minimum(ev_t[np.minimum(nn, max(n - 1, 0))] - idx if n else CAP, CAP), CAP).astype(np.float64)
    # run of consecutive same-team touches ending at the last event (passes completed)
    run = np.zeros(n)
    for e in range(n):
        run[e] = run[e - 1] + 1 if e and ev_s[e] == ev_s[e - 1] else 1
    new["same_team_run"] = np.where(j >= 0, np.minimum(run[np.maximum(j, 0)] if n else 0, RUN_CAP), 0) * new["last1_team"]
    # per-team clocks and counts
    for side, s in (("h", 1), ("a", -1)):
        ev = np.zeros(T)
        ev[ev_t[ev_s == s]] = 1
        at = np.maximum.accumulate(np.where(ev > 0, idx, -1))
        nx = np.minimum.accumulate(np.where(ev > 0, idx, T)[::-1])[::-1]
        new["since_touch_" + side] = np.where(at >= 0, np.minimum(idx - at, CAP), CAP).astype(np.float64)
        new["until_touch_" + side] = np.where(nx < T, np.minimum(nx - idx, CAP), CAP).astype(np.float64)
        cs = np.concatenate(([0.0], np.cumsum(ev)))
        for w in COUNT_WIN:
            new["touches_{}_p{}".format(side, w)] = cs[idx + 1] - cs[np.clip(idx - w + 1, 0, T)]
            new["touches_{}_f{}".format(side, w)] = cs[np.clip(idx + w + 1, 0, T)] - cs[idx + 1]
    for w in COUNT_WIN:
        new["touch_balance_c{}".format(w)] = (new["touches_h_p{}".format(w)] + new["touches_h_f{}".format(w)]
                                             - new["touches_a_p{}".format(w)] - new["touches_a_f{}".format(w)])
    return new, []


def _rolling_min(x, w):
    big = 1e6
    y = minimum_filter1d(np.where(np.isnan(x), big, x), size=2 * w + 1, mode="nearest")
    return np.where(y >= big, np.nan, y)


def distance_dynamics_block(ctx, f):
    new = {}
    for key in ("d_h", "d_a"):
        for w in (25, 75):
            new["{}_min{}".format(key, w)] = _rolling_min(f[key], w)
    near_h = np.where(ctx["seen"], (f["d_h"] < f["d_a"]).astype(np.float64), np.nan)
    for w in (25, 75, 250):
        new["home_nearer_c{}".format(w)] = F._roll(near_h, -w, w)
    sp = f["ball_speed"]
    for w in (25, 75):
        m1, m2 = F._roll(sp, -w, w), F._roll(sp * sp, -w, w)
        new["ball_speed_std{}".format(w)] = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    new["ball_speed_max25"] = np.where(np.isnan(sp), np.nan,
                                       maximum_filter1d(np.nan_to_num(sp), size=51, mode="nearest"))
    return new, []


def trend_context_block(ctx, f):
    new = {}
    for key in ("d_diff", "prog_diff", "spread_diff", "v_all", "vprog_h", "vprog_a"):
        x = f[key]
        new[key + "_trend100"] = F._roll(x, 1, 100) - F._roll(x, -100, -1)
        for w in (125, 250):
            new["{}_c{}".format(key, w)] = F._roll(x, -w, w)
    return new, []


SEQ_BLOCKS = (touch_sequence_block, distance_dynamics_block, trend_context_block)


def build_features_v2(m):
    return F.build_features(m, F.V1_BLOCKS + SEQ_BLOCKS)


build_features_v2.cache_key = "v2:{}:{}".format(F.MODULE_KEY, F.cache.source(*SEQ_BLOCKS, _touch_events,
                                                                                _rolling_min))
F.FEATURE_SETS["v2"] = build_features_v2
