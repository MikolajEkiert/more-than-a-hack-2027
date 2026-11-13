"""Stage 1, ball_state: is the ball in play?

Contract: method(match, **params) -> p_alive, float32 (n,), values in [0, 1], no NaN.
1.0 means certainly alive, 0.0 certainly dead. Frames outside the graded halves may hold anything.
"""

import numpy as np

from possession import features
from possession.match import Match
from possession.registry import register


@register("ball_state", "ball_visible")
def ball_visible(match: Match) -> np.ndarray:
    """Alive exactly where the tracking sees the ball (the reference baseline's rule)."""
    return match.ball_seen.astype(np.float32)


def _log_normal(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


@register("ball_state", "stoppage_evidence")
def stoppage_evidence(
    match: Match,
    seen_alive: float = 0.90,
    seen_dead: float = 0.20,
    speed_alive_m_s: float = 3.5,
    speed_alive_sd: float = 1.0,
    speed_dead_m_s: float = 1.6,
    speed_dead_sd: float = 0.6,
    speed_window_s: float = 1.0,
    speed_weight: float = 1.0,
    outside_margin_m: float = 0.5,
    outside_llr: float = -3.0,
    still_m: float = 0.5,
    still_window_s: float = 1.0,
    still_llr: float = -1.5,
) -> np.ndarray:
    """Per-frame evidence that play is live, from ball visibility, players' speed and ball place.

    Naive Bayes over cues that need no labels, returned as a probability (the decoder takes its
    log-odds, so nothing here is smoothed over time beyond the short feature windows):
      * the ball is tracked (`seen_alive` vs `seen_dead`): on this tracking the ball is lost for
        whole stretches of 5 s or more, most of them stoppages
      * the 90th percentile of the players' speed, averaged over `speed_window_s`: someone is
        always running during play, almost nobody during a stoppage (two normal distributions)
      * the ball is tracked beyond the pitch boundary by `outside_margin_m`: Law 9
      * the ball is tracked but has moved less than `still_m` in `still_window_s`: placed for a
        restart
    Defaults are football priors and round numbers, not fitted to any labelled data.
    """
    n, fps = match.n_frames, match.fps
    seen = match.ball_seen
    llr = np.where(seen, np.log(seen_alive / seen_dead), np.log((1 - seen_alive) / (1 - seen_dead)))

    speed = features.collective_speed(match)
    window = max(int(round(speed_window_s * fps)), 1)
    known = ~np.isnan(speed)
    kernel = np.ones(window)
    total = np.convolve(np.where(known, speed, 0.0), kernel, mode="same")
    count = np.convolve(known.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = total / count
    speed_llr = _log_normal(smooth, speed_alive_m_s, speed_alive_sd) - _log_normal(
        smooth, speed_dead_m_s, speed_dead_sd
    )
    # Beyond the faster mode the two-normal ratio would turn around; more speed is never less live.
    cap = _log_normal(speed_alive_m_s, speed_alive_m_s, speed_alive_sd) - _log_normal(
        speed_alive_m_s, speed_dead_m_s, speed_dead_sd
    )
    speed_llr = np.where(smooth > speed_alive_m_s, cap, speed_llr)
    floor = _log_normal(speed_dead_m_s, speed_alive_m_s, speed_alive_sd) - _log_normal(
        speed_dead_m_s, speed_dead_m_s, speed_dead_sd
    )
    speed_llr = np.where(smooth < speed_dead_m_s, floor, speed_llr)
    llr = llr + speed_weight * np.where(count > 0, speed_llr, 0.0)

    edge = features.ball_edge_distance(match)
    with np.errstate(invalid="ignore"):
        llr = llr + np.where(seen & (edge < -outside_margin_m), outside_llr, 0.0)

    xy = match.ball[:, :2]
    lag = max(int(round(still_window_s * fps)), 1)
    moved = np.full(n, np.nan)
    moved[lag:] = np.hypot(xy[lag:, 0] - xy[:-lag, 0], xy[lag:, 1] - xy[:-lag, 1])
    with np.errstate(invalid="ignore"):
        llr = llr + np.where(moved < still_m, still_llr, 0.0)

    return (1.0 / (1.0 + np.exp(-np.clip(llr, -30, 30)))).astype(np.float32)
