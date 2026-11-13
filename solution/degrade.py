"""Make clean external tracking look like the provided BAR-ATH ATD, and measure the ATD's noise.

IDSSE is corrected tracking: the ball is present on 100% of frames and every player is tracked.
The ATD is automatic: the ball is absent on 41% of graded frames, ~20.5 of 22 players are visible,
and the gaps are bursty and concentrated where the play is (crowds, fast or aerial balls). A model
trained on clean data and run on the ATD would meet a regime it has never seen, so training AND
validation both run on degraded copies.

The defaults are the published ATD figures (DATA.md). Run atd_noise_stats() on the BAR-ATH
container once the data is unpacked and replace the burst lengths / context weights with the
measured ones -- that is statistics of the inputs, not of any label, so it is allowed.
"""

import numpy as np

from solution import container as C

# Params
DEFAULTS = {
    "ball_missing": 0.41,          # DATA.md: ball absent on 41% of graded frames
    "ball_run_s": 1.5,             # mean missing-ball burst, seconds (placeholder until measured)
    "w_crowd": 2.0,                # extra hazard when >= 3 players within 3 m of the ball
    "w_fast": 2.0,                 # extra hazard when the ball moves faster than 15 m/s
    "w_off": 1.0,                  # extra hazard when the ball is outside the lines
    "player_visible": 20.5 / 22,   # DATA.md: 20.5 tracked objects per graded frame on average
    "player_run_s": 2.0,           # mean player dropout burst, seconds
    "pos_sigma": 0.25,             # metres of position error, players
    "pos_tau_s": 3.0,              # ...drifting over ~3 s, not redrawn every frame (diagnose_shift:
                                   #    1 s still inflates speeds beyond BAR-ATH's; 3 s matches)
    "ball_sigma": 0.25,            # metres of position error, ball
    "ball_tau_s": 0.4,
}


def _drift(shape, sigma, tau_frames, rng):
    """Zero-mean AR(1) error with std `sigma` and correlation time `tau_frames` along axis 0.

    Tracking error is mostly slow drift (calibration, bounding-box offset). White noise of the
    same size would add ~5 m/s of fake speed to every player -- the bug that made a model
    trained on degraded IDSSE read BAR-ATH's real, smooth motion as everybody standing still.
    """
    from scipy.signal import lfilter
    phi = np.exp(-1.0 / max(tau_frames, 1e-6))
    white = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), shape)
    white[0] = rng.normal(0.0, sigma, shape[1:]) if len(shape) > 1 else rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -phi], white, axis=0).astype(np.float32)


def _run_mask(u_start, u_len, prob, mean_len):
    """Boolean mask of bursts: a burst starts at t with probability prob[t], geometric length."""
    q = 1.0 / max(mean_len, 1.0)
    lengths = np.ceil(np.log(u_len) / np.log1p(-q)).astype(np.int64) if q < 1 else np.ones_like(u_len, np.int64)
    idx = np.arange(len(prob))
    ends = np.where(u_start < prob, idx + lengths, -1)
    return np.maximum.accumulate(ends) > idx


def _calibrated_mask(weight, target, mean_len, rng, valid):
    """Bursts whose hazard is proportional to weight, scaled by bisection to hit `target`."""
    u_start, u_len = rng.random(len(weight)), 1.0 - rng.random(len(weight))
    w = weight / max(weight[valid].mean(), 1e-9)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        rate = _run_mask(u_start, u_len, np.clip(mid * w, 0, 1), mean_len)[valid].mean()
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    return _run_mask(u_start, u_len, np.clip(hi * w, 0, 1), mean_len)


def ball_context(m):
    """Per-frame crowd / fast / off-pitch flags from the (clean) positions."""
    L, W = m["pitch"]
    dist = np.hypot(m["px"] - m["bx"][:, None], m["py"] - m["by"][:, None])
    crowd = np.nansum(dist < 3.0, axis=1) >= 3
    speed = np.zeros(len(m["bx"]))
    speed[1:] = np.hypot(np.diff(m["bx"]), np.diff(m["by"])) * C.FPS
    fast = np.nan_to_num(speed) > 15.0
    off = (m["bx"] < 0) | (m["bx"] > L) | (m["by"] < 0) | (m["by"] > W)
    return crowd, fast, off


def degrade(m, seed=0, **overrides):
    """A copy of match `m` with ATD-like ball loss, player dropout and jitter."""
    p = dict(DEFAULTS, **overrides)
    rng = np.random.default_rng(seed)
    out = dict(m)
    play = m["half"] > 0

    crowd, fast, off = ball_context(m)
    weight = 1.0 + p["w_crowd"] * crowd + p["w_fast"] * fast + p["w_off"] * off
    lost = _calibrated_mask(weight, p["ball_missing"], p["ball_run_s"] * C.FPS, rng, play)
    ball_tau, pos_tau = p["ball_tau_s"] * C.FPS, p["pos_tau_s"] * C.FPS
    bx = m["bx"] + _drift(m["bx"].shape, p["ball_sigma"], ball_tau, rng)
    by = m["by"] + _drift(m["by"].shape, p["ball_sigma"], ball_tau, rng)
    bx[lost], by[lost] = np.nan, np.nan

    px = m["px"] + _drift(m["px"].shape, p["pos_sigma"], pos_tau, rng)
    py = m["py"] + _drift(m["py"].shape, p["pos_sigma"], pos_tau, rng)
    for s in range(px.shape[1]):
        present = ~np.isnan(m["px"][:, s]) & play
        if present.sum() < 10:
            continue
        gone = _calibrated_mask(np.ones(len(px)), 1 - p["player_visible"],
                                p["player_run_s"] * C.FPS, rng, present)
        px[gone, s], py[gone, s] = np.nan, np.nan

    out.update(px=px.astype(np.float32), py=py.astype(np.float32),
               bx=bx.astype(np.float32), by=by.astype(np.float32))
    out["name"] = m["name"] + "_deg{}".format(seed)
    return out


def degrade_key(m, seed, params=None):
    """Cache key of degrade(m, seed, **params): code, input, seed and the full parameter set."""
    import sys
    from solution import cache
    full = dict(DEFAULTS, **(params or {}))
    return cache.digest("degrade", cache.source(sys.modules[__name__]), cache.fingerprint(m),
                        seed, sorted(full.items()))


def calibrated_params(m):
    """DEFAULTS overrides measured on a real ATD container (the notebook's section 3)."""
    s = atd_noise_stats(m)
    return {"ball_missing": s["ball_missing"], "ball_run_s": s["ball_run_s_mean"],
            "player_visible": s["players_visible_mean"] / 22}


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def atd_noise_stats(m):
    """The numbers DEFAULTS should be set to, measured on a container (use it on BAR-ATH).

    Burst statistics only -- nothing here reads a label.
    """
    play = m["half"] > 0
    missing = np.isnan(m["bx"]) & play
    runs = _runs(missing) / C.FPS
    seen = ~np.isnan(m["px"][play])
    return {
        "ball_missing": float(missing[play].mean()),
        "ball_run_s_mean": float(runs.mean()) if len(runs) else 0.0,
        "ball_run_s_median": float(np.median(runs)) if len(runs) else 0.0,
        "ball_run_s_p90": float(np.percentile(runs, 90)) if len(runs) else 0.0,
        "players_visible_mean": float(seen.sum(1).mean()),
    }
