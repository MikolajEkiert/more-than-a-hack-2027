"""Config + Match -> labels. The only code path that produces a submission.

Deterministic by construction: no wall clock, no unseeded randomness, no dependence on file
iteration order. Methods that need randomness take an explicit `seed: int = 0` parameter.
RULES.md 3 requires the repository to regenerate the submitted file byte for byte.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession import registry
from possession.config import Config
from possession.match import Match


@dataclass
class Result:
    labels: np.ndarray  # <U1 (n,)
    p_alive: np.ndarray
    p_home: np.ndarray


def run(config: Config, match: Match) -> Result:
    stages = config.stages
    p_alive = _call("ball_state", stages["ball_state"], match)
    _check_prob("ball_state", p_alive, match.n_frames, allow_nan=False)

    p_home = _call("team", stages["team"], match)
    _check_prob("team", p_home, match.n_frames, allow_nan=True)

    labels = _call("decode", stages["decode"], match, p_alive, p_home)
    labels = np.asarray(labels).astype("<U1")
    if labels.shape != (match.n_frames,):
        raise ValueError(f"decode returned shape {labels.shape}, expected ({match.n_frames},)")
    bad = sorted(set(np.unique(labels)) - {"H", "A", "D", "X"})
    if bad:
        raise ValueError(f"decode returned labels outside H/A/D/X: {bad[:5]}")

    labels = labels.copy()
    labels[~match.graded] = "X"  # ungraded frames are never scored; X is the honest answer
    return Result(labels=labels, p_alive=p_alive, p_home=p_home)


def _call(stage, stage_config, *inputs):
    method = registry.get(stage, stage_config.method)
    return method.fn(*inputs, **stage_config.params)


def _check_prob(stage, values, n, allow_nan):
    values = np.asarray(values)
    if values.shape != (n,):
        raise ValueError(f"{stage} returned shape {values.shape}, expected ({n},)")
    finite = values[~np.isnan(values)]
    if not allow_nan and finite.size != n:
        raise ValueError(f"{stage} returned NaN; only the team stage may")
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        raise ValueError(f"{stage} returned values outside [0, 1]")


def write_csv(path: Path, labels: np.ndarray) -> str:
    """Write the submission and return its sha256. Same bytes on every OS (explicit '\\n')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "frame,label\n" + "".join(f"{i},{label}\n" for i, label in enumerate(labels))
    data = body.encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
