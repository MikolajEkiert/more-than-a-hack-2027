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
    moving_m_s: float = 4.0,
    moving_llr: float = 2.0,
    kickoff_llr: float = -8.0,
    kickoff_wait_s: float = 300.0,
    kickoff_wait_speed_m_s: float = 2.5,
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
      * the ball is tracked inside the pitch and moving at `moving_m_s` or more: in play (after a
        stoppage the ball is usually tracked again only once it has been played)
      * a kick-off set-up (`features.kickoffs`: each team in its own half): dead until the kick.
        At the start of a half, the wait before the first set-up is dead too, if that set-up
        begins within `kickoff_wait_s` and the players were not running before it
        (`kickoff_wait_speed_m_s`); a half whose data starts at the kick-off has no such wait
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

    ball_speed = features.ball_kinematics(match)["speed"]
    with np.errstate(invalid="ignore"):
        llr = llr + np.where(seen & (ball_speed >= moving_m_s) & (edge >= 0), moving_llr, 0.0)

    if kickoff_llr:
        waiting = features.kickoff_wait(match, kickoff_wait_s, kickoff_wait_speed_m_s)
        llr = llr + np.where(waiting, kickoff_llr, 0.0)

    return (1.0 / (1.0 + np.exp(-np.clip(llr, -30, 30)))).astype(np.float32)


@register("ball_state", "blend")
def blend(match: Match, sources: list = ("stoppage_evidence",), weights: list = ()) -> np.ndarray:
    """Weighted sum of the log-odds of several ball_state sources (methods or probability files).

    `sources` lists registered methods by name, or tables `{method = "...", <its parameters>}` or
    `{file = "path/{game}.npz"}`; `weights` gives one weight per source (all 1 if empty). See
    docs/FUSION.md. A frame on which no source has a value counts as even odds.
    """
    from possession import fusion

    out = fusion.blend(match, "ball_state", sources, weights)
    return np.where(np.isnan(out), 0.5, out).astype(np.float32)


@register("ball_state", "kickoff_wait")
def kickoff_wait(
    match: Match,
    kickoff_llr: float = -8.0,
    kickoff_wait_s: float = 300.0,
    kickoff_wait_speed_m_s: float = 2.5,
) -> np.ndarray:
    """Dead while play waits for a kick-off (`features.kickoff_wait`), no opinion elsewhere.

    A source for `ball_state.blend`: it adds the formation-based kick-off rule to any other
    evidence of play, for instance a supervised model that has never seen the wait before a
    half begins. Returns the probability with log-odds `kickoff_llr` while waiting, 0.5 otherwise.
    """
    waiting = features.kickoff_wait(match, kickoff_wait_s, kickoff_wait_speed_m_s)
    low = 1.0 / (1.0 + np.exp(-kickoff_llr))
    return np.where(waiting, low, 0.5).astype(np.float32)
