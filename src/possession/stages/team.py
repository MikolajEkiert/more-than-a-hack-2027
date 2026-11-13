"""Stage 2, team: if the ball is alive, which team has it?

Contract: method(match, **params) -> p_home, float32 (n,), values in [0, 1] or NaN.
1.0 means certainly home, 0.0 certainly away, NaN means this method has no evidence on that frame
(the decoder decides what to do with it). Do not encode "dead ball" here; that is ball_state's job.
"""

import numpy as np

from possession import features
from possession.features import nearest_player as nearest_player_feature
from possession.match import HOME, Match
from possession.registry import register


@register("team", "nearest_player")
def nearest_player(match: Match) -> np.ndarray:
    """Team of the tracked player nearest the ball; NaN where there is no ball or no player."""
    _, team = nearest_player_feature(match)
    out = np.full(match.n_frames, np.nan, dtype=np.float32)
    out[team == HOME] = 1.0
    out[(team >= 0) & (team != HOME)] = 0.0
    return out


@register("team", "control_evidence")
def control_evidence(
    match: Match,
    control_radius_m: float = 2.0,
    margin_scale_m: float = 0.5,
    duel_radius_m: float = 1.0,
    carry_speed_m_s: float = 4.0,
    passing_by_weight: float = 0.3,
) -> np.ndarray:
    """Soft evidence of the team in control of a tracked ball; NaN when nobody is close to it.

    Evidence exists only when a player is within `control_radius_m` of the ball. Its direction
    is the difference between the two teams' nearest players (a logistic of the margin in
    `margin_scale_m`); when both teams are within `duel_radius_m` it is a duel and there is none.
    Its strength depends on whether the ball moves with that player: a ball travelling more than
    `carry_speed_m_s` faster or slower than the nearest player is passing by, not under control,
    and counts `passing_by_weight` as much. Possession between these moments is the decoder's
    job (it keeps the last owner), which is the provider's rule for loose and contested balls.
    """
    near = features.nearest_by_team(match)
    d_home, d_away = near["distance"][:, 0], near["distance"][:, 1]
    with np.errstate(invalid="ignore"):
        closest = np.fmin(d_home, d_away)
        margin = np.where(np.isnan(d_away), np.inf, d_away) - np.where(
            np.isnan(d_home), np.inf, d_home
        )
        has = closest < control_radius_m
        duel = (d_home < duel_radius_m) & (d_away < duel_radius_m)
    lean = 1.0 / (1.0 + np.exp(-np.clip(margin / margin_scale_m, -30, 30)))

    # Speed of the ball relative to the nearest player: small when it is being carried or held.
    ball_speed = features.ball_kinematics(match)["speed"]
    home_closer = np.where(np.isnan(d_away), True, d_home <= d_away)
    slot = np.where(home_closer, near["slot"][:, 0], near["slot"][:, 1])
    player_speed = match.players[np.arange(match.n_frames), np.maximum(slot, 0), 2]
    with np.errstate(invalid="ignore"):
        relative = np.abs(ball_speed - player_speed)
        weight = np.where(relative > carry_speed_m_s, passing_by_weight, 1.0)

    out = np.full(match.n_frames, np.nan, dtype=np.float32)
    ok = has & ~duel & ~np.isnan(lean)
    out[ok] = (0.5 + (lean[ok] - 0.5) * weight[ok]).astype(np.float32)
    return out
