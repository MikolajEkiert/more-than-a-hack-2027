"""An extra feature block for the supervised model: which team controls a visible, noisy ball.

Task 14. The submitted model's features (the package's set "v1", solution/features.py) answer
"who has the ball" mostly with the nearest player of each team, counts within 3 and 5 m and
touches read from changes of the ball's direction. On the competition match the ball is tracked
on about 92% of live frames but sits 1.5-2 m off the player who has it, and tracks are anonymous,
so a nearest distance often names the wrong team. This block adds evidence that does not hinge on
one distance at one frame:

  * goal side: the defending team has most of its players between the ball and its own goal;
  * carry: the player who has the ball moves with it, the same player for a while;
  * temporal association: the possessor stays near the ball, a player passing by does not;
  * pressure: defenders converge on the ball, the team in possession spreads;
  * shape against the team's own norm: a team spreads out when it has the ball;
  * where the ball is missing, the goal-side reading of its last and next tracked positions;
  * past, future and centred means of the main ones.

Input is a container of the package's layout (a dict: `px`, `py` float (T, S) metres, `team`
(S,) 0 home / 1 away, `bx`, `by` (T,) NaN where the ball is missing, `half` (T,), `pitch`
(L, W)). Every half is oriented as `solution.features.orient` does it (home attacks +x, so home
defends x = 0 and away x = L). The output is float16, as the package stores its own features,
so a cached block and a fresh one are the same numbers. Vectorised numpy, NaN wherever a
quantity is undefined (LightGBM takes NaN natively), deterministic.

This module does not import the package; `possession.supervised` is the only bridge to it.
"""

import hashlib
import inspect
import json
import sys
import warnings

import numpy as np

FPS = 25  # the package's containers are all at 25 frames per second
PARAMS = {
    "lateral_m": 20.0,  # goal side "near the ball": within this many metres across the pitch
    "velocity_span": 2,  # frames on each side of a centred difference (0.16 s in all)
    "speed_cap_m_s": 12.0,  # a faster track is a slot switch, not a sprint
    "carry_lag_s": 0.4,  # "the same player was nearest this long ago"
    "carry_smooth": 5,  # frames on each side of the centred mean of the carry cues (0.4 s)
    "assoc_half_widths": (5, 15, 30),  # +-0.2, +-0.6, +-1.2 s
    "assoc_min_share": 0.3,  # a player's mean distance needs this share of the window seen
    "press_radius_m": 15.0,
    "press_close_m": 10.0,
    "press_speed_m_s": 1.5,
    "context": (("c", -25, 25), ("p", -25, -1), ("p", -100, -1), ("f", 1, 25), ("f", 1, 100)),
}
# The cues that also get past / future / centred means (`PARAMS["context"]`). The same-track cue
# is not among them: on the training matches its mean hardly differs between the two teams' spells.
CONTEXT_KEYS = (
    "t_gs_diff",
    "t_gsl_diff",
    "t_rv_diff",
    "t_assoc_diff_15",
    "t_press_diff",
    "t_npress_diff",
    "t_shape_area_diff",
)
DTYPE = np.float16  # the package's feature dtype (solution/cache.py FEATURE_DTYPE)


def block_key() -> str:
    """Changes whenever this module or its parameters change: part of every cache key of it."""
    text = inspect.getsource(sys.modules[__name__]) + json.dumps(PARAMS, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def feature_names() -> list[str]:
    """The block's column names, in order (computed on a tiny synthetic container)."""
    n, slots = 300, 4
    grid = np.linspace(10.0, 60.0, n * slots, dtype=np.float32).reshape(n, slots)
    m = {
        "px": grid,
        "py": grid[:, ::-1].copy(),
        "team": np.array([0, 0, 1, 1], dtype=np.int8),
        "bx": grid[:, 0].copy(),
        "by": grid[:, 1].copy(),
        "half": np.ones(n, dtype=np.int8),
        "pitch": np.array([105.0, 68.0], dtype=np.float32),
    }
    return build(m)[1]


# -- helpers ---------------------------------------------------------------------------------


def _quiet(fn, *args, **kwargs):
    """`fn` with numpy's warnings about all-NaN slices and empty means silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def orient(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(px, py, bx, by) float64, each half turned so that home attacks +x.

    The same rule as `solution.features.orient`: a half whose mean home x exceeds the mean away
    x is rotated by 180 degrees (x -> L - x, y -> W - y)."""
    length, width = (float(v) for v in m["pitch"])
    px, py = np.array(m["px"], dtype=np.float64), np.array(m["py"], dtype=np.float64)
    bx, by = np.array(m["bx"], dtype=np.float64), np.array(m["by"], dtype=np.float64)
    team, half = np.asarray(m["team"]), np.asarray(m["half"])
    for h in (1, 2):
        sel = half == h
        if not sel.any():
            continue
        home = _quiet(np.nanmean, px[sel][:, team == 0], axis=1)
        away = _quiet(np.nanmean, px[sel][:, team == 1], axis=1)
        if _quiet(np.nanmean, home) > _quiet(np.nanmean, away):
            px[sel], py[sel] = length - px[sel], width - py[sel]
            bx[sel], by[sel] = length - bx[sel], width - by[sel]
    return px, py, bx, by


def swapped(m: dict) -> dict:
    """The container with home and away exchanged (`solution.container.swap_teams` for the
    fields this block reads): the second orientation the model is trained on."""
    return {**m, "team": (1 - np.asarray(m["team"])).astype(np.int8)}


def rolling_mean(x: np.ndarray, lo: int, hi: int, min_count: int = 1) -> np.ndarray:
    """NaN-aware mean of x[t + lo .. t + hi] (inclusive offsets) along axis 0, by cumulative
    sums; NaN where fewer than `min_count` values are known. Works on (T,) and (T, S)."""
    n = len(x)
    known = ~np.isnan(x)
    zero = np.zeros((1,) + x.shape[1:])
    total = np.concatenate([zero, np.cumsum(np.where(known, x, 0.0), axis=0)])
    count = np.concatenate([zero, np.cumsum(known, axis=0)])
    idx = np.arange(n)
    a, b = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    have = count[b] - count[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[b] - total[a]) / have
    return np.where(have >= max(min_count, 1), mean, np.nan)


def _velocity(x: np.ndarray, k: int) -> np.ndarray:
    """Centred difference over 2k frames, per second, along axis 0; NaN at the ends or where
    either end is missing."""
    out = np.full(x.shape, np.nan)
    if len(x) > 2 * k:
        out[k:-k] = (x[2 * k :] - x[: -2 * k]) * FPS / (2.0 * k)
    return out


def _nearest_slot(dist: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Per frame, the slot of the team (`cols` mask) nearest the ball; -1 if none is known."""
    n = len(dist)
    if not cols.any():
        return np.full(n, -1)
    d = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols])
    j = np.argmin(d, axis=1)
    found = np.isfinite(d[np.arange(n), j])
    return np.where(found, np.flatnonzero(cols)[j], -1)


def _fill_index(seen: np.ndarray, half: np.ndarray, forward: bool) -> np.ndarray:
    """Index of the last (forward) or next tracked-ball frame within the same half; -1 if none."""
    n = len(seen)
    idx = np.arange(n)
    run = np.concatenate([[0], np.cumsum(half[1:] != half[:-1])])
    if forward:
        at = np.maximum.accumulate(np.where(seen, idx, -1))
    else:
        at = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    inside = (at >= 0) & (at < n)
    ok = inside & (run[np.clip(at, 0, n - 1)] == run) & (half > 0)
    return np.where(ok, at, -1)


def _team_norm(x: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(x - its median) / its interquartile range, per half; NaN where undefined."""
    out = np.full(len(x), np.nan)
    for h in (1, 2):
        sel = (half == h) & ~np.isnan(x)
        if sel.sum() < 2:
            continue
        q25, q50, q75 = np.percentile(x[sel], [25, 50, 75])
        if q75 > q25:
            part = half == h
            out[part] = (x[part] - q50) / (q75 - q25)
    return out


# -- the cues --------------------------------------------------------------------------------


def goal_side(px, py, bx, by, home, away, lateral_m):
    """Per team, its visible players between the ball and its own goal (home's at x = 0, away's
    at x = L), counted overall and within `lateral_m` across the pitch, and their share of the
    team's visible players. NaN where the ball (or the team) is not seen."""
    seen = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        lateral = np.abs(py - by[:, None]) <= lateral_m
        behind = {"h": px[:, home] < bx[:, None], "a": px[:, away] > bx[:, None]}
    out = {}
    for side, cols in (("h", home), ("a", away)):
        visible = np.sum(~np.isnan(px[:, cols]), axis=1)
        count = np.sum(behind[side], axis=1).astype(np.float64)
        near = np.sum(behind[side] & lateral[:, cols], axis=1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = count / visible
        out[side] = {
            "n": np.where(seen, count, np.nan),
            "share": np.where(seen & (visible > 0), share, np.nan),
            "near": np.where(seen, near, np.nan),
        }
    return out


def build(m: dict) -> tuple[np.ndarray, list[str]]:
    """(F, names): the block for container `m`, F float16 (T, K). Every feature is documented
    where it is computed; `names` gives the column order."""
    p = PARAMS
    px, py, bx, by = orient(m)
    team, half = np.asarray(m["team"]), np.asarray(m["half"])
    home, away = team == 0, team == 1
    n = len(bx)
    seen = ~np.isnan(bx)
    f = {}

    # -- goal side ----------------------------------------------------------------------------
    gs = goal_side(px, py, bx, by, home, away, p["lateral_m"])
    # t_gs_n_h / _a: number of the team's visible players goal-side of the ball
    # t_gs_share_h / _a: the same as a share of the team's visible players
    # t_gs_diff: home share minus away share (> 0: home defends, so away has the ball); robust to
    #   a ball 1-2 m off, which moves only the players level with it
    # t_gsl_n_h / _a: goal-side players within `lateral_m` across the pitch of the ball
    # t_gsl_diff: home minus away of the latter
    for side in ("h", "a"):
        f[f"t_gs_n_{side}"] = gs[side]["n"]
        f[f"t_gs_share_{side}"] = gs[side]["share"]
        f[f"t_gsl_n_{side}"] = gs[side]["near"]
    f["t_gs_diff"] = gs["h"]["share"] - gs["a"]["share"]
    f["t_gsl_diff"] = gs["h"]["near"] - gs["a"]["near"]

    # t_gs_diff_last / _next: t_gs_diff of the players now, against the last (next) tracked ball
    #   position in the same half: what the arrangement says while the ball is not seen
    for tag, forward in (("last", True), ("next", False)):
        at = _fill_index(seen, half, forward)
        safe = np.maximum(at, 0)
        fx = np.where(at >= 0, bx[safe], np.nan)
        fy = np.where(at >= 0, by[safe], np.nan)
        filled = goal_side(px, py, fx, fy, home, away, p["lateral_m"])
        f[f"t_gs_diff_{tag}"] = filled["h"]["share"] - filled["a"]["share"]

    # -- carry ------------------------------------------------------------------------------
    k = p["velocity_span"]
    dist = np.hypot(px - bx[:, None], py - by[:, None])  # NaN where ball or player unseen
    vpx, vpy = _velocity(px, k), _velocity(py, k)
    speed = np.hypot(vpx, vpy)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(speed > p["speed_cap_m_s"], p["speed_cap_m_s"] / speed, 1.0)
    vpx, vpy = vpx * scale, vpy * scale
    vbx, vby = _velocity(bx, k), _velocity(by, k)
    lag = int(round(p["carry_lag_s"] * FPS))
    w = p["carry_smooth"]
    rows = np.arange(n)
    for side, cols in (("h", home), ("a", away)):
        slot = _nearest_slot(dist, cols)
        has = slot >= 0
        s = np.maximum(slot, 0)
        # t_rv_h / _a: |v_ball - v_player| of the team's player nearest the ball, mean over
        #   +-0.4 s: near zero while he carries it, large when it flies past him
        relative = np.hypot(vbx - vpx[rows, s], vby - vpy[rows, s])
        f[f"t_rv_{side}"] = rolling_mean(np.where(has, relative, np.nan), -w, w)
        # t_same_h / _a: share of the +-0.4 s window on which the team's nearest player is the
        #   same track as 0.4 s earlier (one player keeping the ball, not a crowd around it)
        same = np.full(n, np.nan)
        if n > lag:
            both = has[lag:] & has[:-lag]
            same[lag:] = np.where(both, (slot[lag:] == slot[:-lag]).astype(np.float64), np.nan)
        f[f"t_same_{side}"] = rolling_mean(same, -w, w)
    f["t_rv_diff"] = f["t_rv_h"] - f["t_rv_a"]
    f["t_same_diff"] = f["t_same_h"] - f["t_same_a"]

    # -- temporal association -------------------------------------------------------------------
    # t_assoc_h_<w> / _a_<w>: over the team's players, the smallest mean distance to the ball over
    #   +-w frames (each player's own NaN-aware mean, needing `assoc_min_share` of the window):
    #   the possessor stays near the ball, a player passing by does not
    # t_assoc_diff_<w>: home minus away
    for hw in p["assoc_half_widths"]:
        need = max(2, int(np.ceil(p["assoc_min_share"] * (2 * hw + 1))))
        mean = rolling_mean(dist, -hw, hw, need)
        for side, cols in (("h", home), ("a", away)):
            f[f"t_assoc_{side}_{hw}"] = (
                _quiet(np.nanmin, mean[:, cols], axis=1) if cols.any() else np.full(n, np.nan)
            )
        f[f"t_assoc_diff_{hw}"] = f[f"t_assoc_h_{hw}"] - f[f"t_assoc_a_{hw}"]
        del mean

    # -- pressure -------------------------------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        toward = (vpx * (bx[:, None] - px) + vpy * (by[:, None] - py)) / dist
        toward = np.where(dist > 1e-6, toward, np.nan)  # at the ball: no direction to it
        within = dist < p["press_radius_m"]
        closing = (dist < p["press_close_m"]) & (toward > p["press_speed_m_s"])
    for side, cols in (("h", home), ("a", away)):
        # t_press_h / _a: mean speed toward the ball of the team's players within
        #   `press_radius_m` of it (defenders converge, the team in possession spreads)
        f[f"t_press_{side}"] = _quiet(
            np.nanmean, np.where(within[:, cols], toward[:, cols], np.nan), axis=1
        )
        # t_npress_h / _a: the team's players within `press_close_m` closing in on the ball
        #   faster than `press_speed_m_s`
        f[f"t_npress_{side}"] = np.where(
            seen, np.sum(closing[:, cols], axis=1).astype(np.float64), np.nan
        )
    f["t_press_diff"] = f["t_press_h"] - f["t_press_a"]
    f["t_npress_diff"] = f["t_npress_h"] - f["t_npress_a"]
    del dist, toward, within, closing

    # -- shape against the team's own norm ------------------------------------------------------
    # t_shape_<q>_h / _a: the team's spread q (standard deviation of x "sx", of y "sy", and their
    #   product "area") minus its median over the half, over its interquartile range in the half:
    #   a team spreads out when it has the ball, whatever its usual shape
    # t_shape_area_diff: home minus away of the area
    for side, cols in (("h", home), ("a", away)):
        sx = _quiet(np.nanstd, px[:, cols], axis=1)
        sy = _quiet(np.nanstd, py[:, cols], axis=1)
        f[f"t_shape_sx_{side}"] = _team_norm(sx, half)
        f[f"t_shape_sy_{side}"] = _team_norm(sy, half)
        f[f"t_shape_area_{side}"] = _team_norm(sx * sy, half)
    f["t_shape_area_diff"] = f["t_shape_area_h"] - f["t_shape_area_a"]

    # -- temporal context ---------------------------------------------------------------------
    # <key>_c25: mean over +-1 s; <key>_p25 / _p100: over the past 1 s / 4 s; <key>_f25 / _f100:
    #   over the next 1 s / 4 s (the package's `_roll` windows), for each of CONTEXT_KEYS
    for key in CONTEXT_KEYS:
        for tag, lo, hi in p["context"]:
            f[f"{key}_{tag}{max(abs(lo), abs(hi))}"] = rolling_mean(f[key], lo, hi)

    names = list(f)
    out = np.column_stack([np.asarray(f[k], dtype=np.float64) for k in names])
    return out.astype(np.float32).astype(DTYPE), names
