"""Combining several sources of evidence for one stage (docs/FUSION.md).

A source is a registered method of the stage, or a file of per-frame probabilities produced by a
model that lives outside the registry. Sources are combined as a weighted sum of log-odds, the
same quantity the decoder consumes, so two tracks (for instance the label-free evidence and a
supervised model) are fused as evidence, before decoding. The decoder then still produces whole
possessions; voting on the two tracks' labels frame by frame would bring the flicker back.
"""

from pathlib import Path

import numpy as np

from possession import registry
from possession._paths import REPO_ROOT
from possession.match import Match

EPS = 1e-6
FILE_KEYS = {"ball_state": "p_alive", "team": "p_home"}


def class_evidence(stage: str, p_h, p_a, p_d) -> np.ndarray:
    """A stage's evidence from three class probabilities per frame, P(H), P(A), P(D):
    p_alive = 1 - P(D) for `ball_state`, p_home = P(H) / (P(H) + P(A)) for `team` (NaN where the
    model gives the two teams no probability at all)."""
    p_h, p_a, p_d = (np.asarray(p, dtype=np.float64) for p in (p_h, p_a, p_d))
    if stage == "ball_state":
        return 1.0 - p_d
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p_h + p_a > EPS, p_h / (p_h + p_a), np.nan)


def from_file(match: Match, stage: str, path: str, key: str | None = None) -> np.ndarray:
    """Per-frame evidence for `stage` read from a .npy or .npz file.

    `{game}` in `path` is replaced by `match.game` (for a degraded external match that is
    "<GAME>@<profile>#<seed>", so every validation condition has its own file). Relative paths are
    relative to the repository. Accepted contents:
      * .npy: the stage's own array, (n_frames,)
      * .npz with `p_alive` / `p_home` (or the name given as `key`)
      * .npz with `p_h`, `p_a`, `p_d`: three class probabilities per frame, from which
        p_alive = 1 - p_d and p_home = p_h / (p_h + p_a)
    """
    file = Path(path.format(game=match.game))
    if not file.is_absolute():
        file = REPO_ROOT / file
    if not file.is_file():
        raise FileNotFoundError(f"{stage} evidence file not found: {file}")
    if file.suffix == ".npy":
        values = np.load(file, allow_pickle=False)
    else:
        with np.load(file, allow_pickle=False) as data:
            name = key or FILE_KEYS[stage]
            if name in data:
                values = data[name]
            elif {"p_h", "p_a", "p_d"} <= set(data.files):
                values = class_evidence(stage, data["p_h"], data["p_a"], data["p_d"])
            else:
                raise KeyError(f"{file.name}: no '{name}' and no p_h/p_a/p_d in {data.files}")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (match.n_frames,):
        raise ValueError(f"{file.name}: shape {values.shape}, expected ({match.n_frames},)")
    finite = values[~np.isnan(values)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        raise ValueError(f"{file.name}: values outside [0, 1]")
    return values


def evidence(match: Match, stage: str, source) -> np.ndarray:
    """One source's (n,) array. `source` is a method name, or a table with `method` (plus that
    method's parameters) or with `file` (plus an optional `key`)."""
    if isinstance(source, str):
        return np.asarray(registry.get(stage, source).fn(match), dtype=np.float64)
    spec = dict(source)
    if "file" in spec:
        unknown = sorted(set(spec) - {"file", "key"})
        if unknown:
            raise ValueError(f"file source takes 'file' and 'key', not {unknown}")
        return from_file(match, stage, spec["file"], spec.get("key"))
    if "method" not in spec:
        raise ValueError(f"a source needs 'method' or 'file': {source!r}")
    method = registry.get(stage, spec.pop("method"))
    unknown = sorted(set(spec) - set(method.params))
    if unknown:
        raise ValueError(f"{stage}.{method.name} has no parameter(s) {unknown}")
    return np.asarray(method.fn(match, **spec), dtype=np.float64)


def combine(values: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted sum of log-odds, back as a probability. NaN means "this source has no vote";
    where every source is NaN the result is NaN."""
    total = np.zeros(len(values[0]))
    voted = np.zeros(len(values[0]), dtype=bool)
    for value, weight in zip(values, weights, strict=True):
        known = ~np.isnan(value)
        clipped = np.clip(np.where(known, value, 0.5), EPS, 1 - EPS)
        total += np.where(known, weight * (np.log(clipped) - np.log1p(-clipped)), 0.0)
        voted |= known
    out = 1.0 / (1.0 + np.exp(-np.clip(total, -30, 30)))
    return np.where(voted, out, np.nan)


def blend(match: Match, stage: str, sources, weights) -> np.ndarray:
    """Evidence of `stage` from all `sources`, combined with `weights` (all 1 when empty)."""
    sources = list(sources)
    if not sources:
        raise ValueError(f"{stage}.blend needs at least one source")
    weights = [1.0] * len(sources) if not len(weights) else [float(w) for w in weights]
    if len(weights) != len(sources):
        raise ValueError(f"{len(sources)} sources but {len(weights)} weights")
    return combine([evidence(match, stage, source) for source in sources], weights)
