"""Stage 1, ball_state: is the ball in play?

Contract: method(match, **params) -> p_alive, float32 (n,), values in [0, 1], no NaN.
1.0 means certainly alive, 0.0 certainly dead. Frames outside the graded halves may hold anything.
"""

import numpy as np

from possession.match import Match
from possession.registry import register


@register("ball_state", "ball_visible")
def ball_visible(match: Match) -> np.ndarray:
    """Alive exactly where the tracking sees the ball (the reference baseline's rule)."""
    return match.ball_seen.astype(np.float32)
