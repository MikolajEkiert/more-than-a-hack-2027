"""v0 per-frame features, identical for IDSSE and BAR-ATH containers.

Only what both sources have: anonymous player slots with a team, and a ball (x, y) that may be
missing. No person ids, no ball height, no Smart Tagging -- those are target-only and are added
later (see the notebook plan), never here, so the model trained on IDSSE can run on BAR-ATH.

Everything is in metres and oriented per half so that the home team always attacks +x. LightGBM
takes NaN natively, so a missing ball is left as NaN rather than imputed.
"""

import sys
import warnings

import numpy as np

from solution import cache, container as C

# Params
SPEED_CAP = 12.0                 # m/s; slot identity switches produce impossible jumps
CLOSE = 3.0                      # metres, "near the ball"
CAP = 750                        # frames, cap for since/until-seen counters (30 s)
CENTRED = (5, 25, 75)            # half-widths of centred windows, frames (0.2 s, 1 s, 3 s)
SIDE = (25, 100)                 # past-only / future-only windows, frames (1 s, 4 s)
TEMPORAL = ("ball_seen", "d_diff", "ball_speed", "ball_out", "v_all", "prog_diff",
            "vprog_h", "vprog_a", "n_h_close", "n_a_close")


def _nanreduce(x, cols, fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(x[:, cols], axis=1)


def _roll(x, lo, hi):
    """NaN-aware mean of x[t+lo .. t+hi] (inclusive offsets), via cumulative sums."""
    T = len(x)
    ok = ~np.isnan(x)
    cv = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cc = np.concatenate(([0.0], np.cumsum(ok)))
    idx = np.arange(T)
    a, b = np.clip(idx + lo, 0, T), np.clip(idx + hi + 1, 0, T)
    n = cc[b] - cc[a]
    out = np.full(T, np.nan)
    good = n > 0
    out[good] = (cv[b] - cv[a])[good] / n[good]
    return out


def _diff(x, k):
    """Centred difference over 2k frames, per second. NaN where either end is missing."""
    out = np.full(x.shape, np.nan)
    out[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k / C.FPS)
    return out


def orient(m):
    """Positions mirrored per half so the home team attacks +x."""
    L, W = m["pitch"]
    px, py = m["px"].astype(np.float64), m["py"].astype(np.float64)
    bx, by = m["bx"].astype(np.float64), m["by"].astype(np.float64)
    for h in (1, 2):
        sel = m["half"] == h
        if not sel.any():
            continue
        home = _nanreduce(px[sel], m["team"] == 0, np.nanmean)
        away = _nanreduce(px[sel], m["team"] == 1, np.nanmean)
        if np.nanmean(home) > np.nanmean(away):
            px[sel], py[sel] = L - px[sel], W - py[sel]
            bx[sel], by[sel] = L - bx[sel], W - by[sel]
    return px, py, bx, by


def build_features(m, blocks=()):
    """(F, names): float32 (T, K) feature matrix for one container.

    `blocks` are extra feature functions block(ctx, f) -> (dict of new series, temporal keys);
    their series get the same centred / past / future windows as the v0 ones listed in TEMPORAL.
    """
    L, W = [float(v) for v in m["pitch"]]
    px, py, bx, by = orient(m)
    home, away = m["team"] == 0, m["team"] == 1
    T = len(bx)
    f = {}
    temporal = list(TEMPORAL)

    # -- ball -------------------------------------------------------------------------------
    seen = ~np.isnan(bx)
    f["ball_seen"] = seen.astype(np.float64)
    f["bx"], f["by_c"] = bx, np.abs(by - W / 2)
    vx, vy = _diff(bx, 1), _diff(by, 1)
    f["ball_speed"] = np.hypot(vx, vy)
    f["ball_vx"] = vx
    f["ball_out"] = np.maximum.reduce([np.zeros(T), -bx, bx - L, -by, by - W])
    f["ball_out"][~seen] = np.nan
    f["ball_edge"] = np.minimum.reduce([bx, L - bx, by, W - by])

    # -- player <-> ball --------------------------------------------------------------------
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    d_h = _nanreduce(dist, home, np.nanmin)
    d_a = _nanreduce(dist, away, np.nanmin)
    f["d_h"], f["d_a"], f["d_diff"] = d_h, d_a, d_h - d_a
    f["n_h_close"] = np.where(seen, np.nansum(dist[:, home] < CLOSE, 1), np.nan)
    f["n_a_close"] = np.where(seen, np.nansum(dist[:, away] < CLOSE, 1), np.nan)

    # -- players only (available when the ball is not) ----------------------------------------
    speed = np.minimum(np.hypot(_diff(px, 2), _diff(py, 2)), SPEED_CAP)
    f["n_vis_h"] = np.sum(~np.isnan(px[:, home]), 1).astype(np.float64)
    f["n_vis_a"] = np.sum(~np.isnan(px[:, away]), 1).astype(np.float64)
    f["v_h"] = _nanreduce(speed, home, np.nanmean)
    f["v_a"] = _nanreduce(speed, away, np.nanmean)
    f["v_all"] = _nanreduce(speed, home | away, np.nanmean)
    f["v_max"] = _nanreduce(speed, home | away, np.nanmax)
    cx_h, cx_a = _nanreduce(px, home, np.nanmean), _nanreduce(px, away, np.nanmean)
    f["prog_h"], f["prog_a"] = cx_h, L - cx_a          # distance advanced toward own target
    f["prog_diff"] = f["prog_h"] - f["prog_a"]
    f["vprog_h"], f["vprog_a"] = _diff(cx_h, 6), -_diff(cx_a, 6)
    for side, cols in (("h", home), ("a", away)):
        f["sx_" + side] = _nanreduce(px, cols, np.nanstd)
        f["sy_" + side] = _nanreduce(py, cols, np.nanstd)
        f["len_" + side] = _nanreduce(px, cols, np.nanmax) - _nanreduce(px, cols, np.nanmin)
        f["wid_" + side] = _nanreduce(py, cols, np.nanmax) - _nanreduce(py, cols, np.nanmin)
    f["spread_diff"] = (f["sx_h"] * f["sy_h"]) - (f["sx_a"] * f["sy_a"])

    # -- occlusion context: what the ball said last time and next time it was seen ------------
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, T)[::-1])[::-1]
    f["since_seen"] = np.where(last >= 0, np.minimum(idx - last, CAP), CAP).astype(np.float64)
    f["until_seen"] = np.where(nxt < T, np.minimum(nxt - idx, CAP), CAP).astype(np.float64)
    for key in ("d_diff", "ball_out", "bx", "ball_speed"):
        f["last_" + key] = np.where(last >= 0, f[key][np.maximum(last, 0)], np.nan)
        f["next_" + key] = np.where(nxt < T, f[key][np.minimum(nxt, T - 1)], np.nan)

    # -- extra blocks (v1+) -----------------------------------------------------------------
    ctx = {"m": m, "L": L, "W": W, "px": px, "py": py, "bx": bx, "by": by, "home": home,
           "away": away, "seen": seen, "dist": dist, "last": last, "nxt": nxt, "T": T}
    for block in blocks:
        new, keys = block(ctx, f)
        f.update(new)
        temporal.extend(keys)

    # -- temporal context -------------------------------------------------------------------
    for key in temporal:
        x = f[key]
        for w in CENTRED:
            f["{}_c{}".format(key, w)] = _roll(x, -w, w)
        for w in SIDE:
            f["{}_p{}".format(key, w)] = _roll(x, -w, -1)
            f["{}_f{}".format(key, w)] = _roll(x, 1, w)

    names = list(f)
    F = np.column_stack([np.asarray(f[k], np.float64) for k in names]).astype(np.float32)
    return F, names


# =============================================================================================
# v1 blocks. Each is one idea, evaluated on its own with the harness before it is kept.
# =============================================================================================

def _ffill_sign(sign, T):
    """Last non-zero value of `sign` at or before t, and frames since it (capped)."""
    idx = np.arange(T)
    at = np.maximum.accumulate(np.where(sign != 0, idx, -1))
    val = np.where(at >= 0, sign[np.maximum(at, 0)], 0).astype(np.float64)
    return val, np.where(at >= 0, np.minimum(idx - at, CAP), CAP).astype(np.float64)


def _bfill_sign(sign, T):
    val, gap = _ffill_sign(sign[::-1], T)
    return val[::-1], gap[::-1]


def contact_block(ctx, f, k=5, radius=1.5):
    """Vidal-Codina's gain rule: the ball changes direction or speed next to a player.

    A touch is where control can change hands; which team touched last, and who touches next,
    is the strongest single answer to "who has it" through an occlusion.
    """
    bx, by, T = ctx["bx"], ctx["by"], ctx["T"]
    vin_x, vin_y = np.full(T, np.nan), np.full(T, np.nan)
    vout_x, vout_y = np.full(T, np.nan), np.full(T, np.nan)
    vin_x[k:], vin_y[k:] = (bx[k:] - bx[:-k]) * C.FPS / k, (by[k:] - by[:-k]) * C.FPS / k
    vout_x[:-k], vout_y[:-k] = (bx[k:] - bx[:-k]) * C.FPS / k, (by[k:] - by[:-k]) * C.FPS / k
    sin, sout = np.hypot(vin_x, vin_y), np.hypot(vout_x, vout_y)
    cos = (vin_x * vout_x + vin_y * vout_y) / np.maximum(sin * sout, 1e-6)
    dv = sout - sin
    d_min = np.fmin(f["d_h"], f["d_a"])
    near = np.where(f["d_h"] <= f["d_a"], 1, -1)
    moving = np.fmax(sin, sout) > 3.0
    touch = (d_min < radius) & moving & ((cos < 0.5) | (np.abs(dv) > 4.0))
    sign = np.where(touch, near, 0)
    last_val, since = _ffill_sign(sign, T)
    next_val, until = _bfill_sign(sign, T)
    new = {
        "ball_cos": cos, "ball_dv": dv,
        "touch_h": np.where(ctx["seen"], (sign == 1).astype(float), np.nan),
        "touch_a": np.where(ctx["seen"], (sign == -1).astype(float), np.nan),
        "last_touch": last_val, "since_touch": since,
        "next_touch": next_val, "until_touch": until,
    }
    return new, ["touch_h", "touch_a", "last_touch", "next_touch"]


def streak_block(ctx, f, cap=250):
    """Signed length of the current run of frames with the same team nearest the ball."""
    T, seen = ctx["T"], ctx["seen"]
    near = np.where(seen, np.where(f["d_h"] <= f["d_a"], 1, -1), 0)
    brk = np.concatenate(([True], near[1:] != near[:-1]))
    start = np.maximum.accumulate(np.where(brk, np.arange(T), 0))
    pos = np.minimum(np.arange(T) - start + 1, cap)
    return {"near_streak": np.where(seen, near * pos, np.nan).astype(np.float64)}, ["near_streak"]


def deadball_block(ctx, f):
    """Restart layouts (Vidal-Codina triggers) and where the ball was heading when lost."""
    L, W, px, py, T = ctx["L"], ctx["W"], ctx["px"], ctx["py"], ctx["T"]
    home, away = ctx["home"], ctx["away"]
    out = (px < 0) | (px > L) | (py < 0) | (py > W)
    corners = [(0, 0), (0, W), (L, 0), (L, W)]
    corner = np.min([_nanreduce(np.hypot(px - cx, py - cy), home | away, np.nanmin)
                     for cx, cy in corners], axis=0)
    centre = _nanreduce(np.hypot(px - L / 2, py - W / 2), home | away, np.nanmin)
    own = (np.nansum(px[:, home] < L / 2, 1) + np.nansum(px[:, away] > L / 2, 1))
    vis = np.sum(~np.isnan(px[:, home | away]), 1)
    # outward speed toward the nearest line at each seen frame, then carried to the gaps
    bx, by = ctx["bx"], ctx["by"]
    vx, vy = _diff(bx, 2), _diff(by, 2)
    edges = np.stack([bx, L - bx, by, W - by])
    nearest = np.nanargmin(np.where(np.isnan(edges), np.inf, edges), axis=0)
    outward = np.choose(nearest, [-vx, vx, -vy, vy])
    last, nxt = ctx["last"], ctx["nxt"]
    new = {
        "n_out": np.nansum(out, 1).astype(np.float64),
        "corner_min": corner, "centre_min": centre,
        "own_half_frac": np.where(vis > 0, own / np.maximum(vis, 1), np.nan),
        "last_outward": np.where(last >= 0, outward[np.maximum(last, 0)], np.nan),
        "last_edge": np.where(last >= 0, f["ball_edge"][np.maximum(last, 0)], np.nan),
        "next_edge": np.where(nxt < T, f["ball_edge"][np.minimum(nxt, T - 1)], np.nan),
        "v_all_c250": _roll(f["v_all"], -250, 250),
        "v_all_p250": _roll(f["v_all"], -250, -1),
        "v_all_f250": _roll(f["v_all"], 1, 250),
    }
    return new, ["n_out", "corner_min", "own_half_frac"]


def pressure_block(ctx, f, r=5.0):
    """How many of each team are within 5 m of the ball."""
    dist, seen = ctx["dist"], ctx["seen"]
    return ({"n_h5": np.where(seen, np.nansum(dist[:, ctx["home"]] < r, 1), np.nan),
             "n_a5": np.where(seen, np.nansum(dist[:, ctx["away"]] < r, 1), np.nan)},
            ["n_h5", "n_a5"])


V1_BLOCKS = (contact_block, streak_block, deadball_block, pressure_block)


def build_features_v1(m):
    return build_features(m, V1_BLOCKS)


def with_blocks(*blocks):
    """Feature function = v0 + the given blocks. Use it to test one idea at a time."""
    fn = lambda m: build_features(m, blocks)  # noqa: E731
    block_src = cache.source(*blocks)
    fn.cache_key = None if block_src is None else "blocks:{}:{}".format(MODULE_KEY, block_src)
    return fn


FEATURE_SETS = {"v0": build_features, "v1": build_features_v1}

# -- caching ------------------------------------------------------------------------------------
# A feature set's key is the hash of this whole file, so editing anything here recomputes it.
MODULE_KEY = cache.source(sys.modules[__name__])
build_features.cache_key = "v0:{}".format(MODULE_KEY)
build_features_v1.cache_key = "v1:{}".format(MODULE_KEY)


def feature_key(fn):
    """Cache key of a feature function: its cache_key, else its source; None = do not cache."""
    key = getattr(fn, "cache_key", None)
    if key is not None:
        return key
    src = cache.source(fn)
    return None if src is None else "fn:{}:{}".format(MODULE_KEY, src)


def _head(m, n=600):
    out = dict(m)
    for k in ("px", "py", "bx", "by", "half", "label"):
        out[k] = m[k][:n]
    return out


def cached(fn, m, tag, match_key):
    """(F float16, names) for match m through the cache. match_key identifies m's exact content."""
    fkey = feature_key(fn)
    names = cache.value("feature_names", cache.digest(fkey), lambda: fn(_head(m))[1])
    F = cache.array("features", cache.digest(fkey, match_key, tag),
                    lambda: fn(m)[0].astype(cache.FEATURE_DTYPE))
    return F, names
