"""Stage 3, decode: per-frame probabilities -> labels.

Contract: method(match, p_alive, p_home, **params) -> labels, <U1 (n,), each in {H, A, D, X}.
The pipeline overwrites ungraded frames with X afterwards, so decoders need not handle them.

This is where possession *structure* lives: spellF1 rewards whole possessions, so smoothing,
minimum durations and state models belong here. Remember the X rule (EVALUATION.md): the scorer
forward-fills X, so X is free inside a possession and expensive across a hand-over.
"""

import numpy as np

from possession.match import Match
from possession.registry import register
from starter.baseline import AWAY, HOME, NOBODY, smooth


def hard_codes(p_alive: np.ndarray, p_home: np.ndarray) -> np.ndarray:
    """int8 codes as starter/baseline.py uses them: HOME, AWAY, or NOBODY (dead / no evidence)."""
    codes = np.full(p_alive.shape[0], NOBODY, dtype=np.int8)
    alive = (p_alive >= 0.5) & ~np.isnan(p_home)
    codes[alive & (p_home >= 0.5)] = HOME
    codes[alive & (p_home < 0.5)] = AWAY
    return codes


@register("decode", "majority_vote")
def majority_vote(match: Match, p_alive, p_home, window: int = 50) -> np.ndarray:
    """Centred majority vote over +/- window frames (the reference baseline's smoother)."""
    return smooth(hard_codes(p_alive, p_home), window).astype("<U1")
