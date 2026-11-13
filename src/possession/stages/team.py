"""Stage 2, team: if the ball is alive, which team has it?

Contract: method(match, **params) -> p_home, float32 (n,), values in [0, 1] or NaN.
1.0 means certainly home, 0.0 certainly away, NaN means this method has no evidence on that frame
(the decoder decides what to do with it). Do not encode "dead ball" here; that is ball_state's job.
"""

import numpy as np

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
