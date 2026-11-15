"""The possession structure of a match, window by window, and weights that match it (task 15).

The official score counts the possessions of one match. How dense its stoppages are decides how many
of its possessions start or end at a stoppage, and so which errors weigh most: the competition match
has about 12 stoppages per 10 graded minutes by version 4's labels (which count stoppages within
about 0.5 of the truth on every validation set without a still ball), every validation truth
9.6-10.7. These
descriptors are computed the same way from the truth of another match and from our own labels of the
competition match (aggregates only), on the 10-minute windows the validation already scores
(`validation.windows`), so a validation window can be weighted by how often the competition match
has windows like it.
"""

import numpy as np

import possession_metrics as pm
from possession import validation

# Stoppages per 10 graded minutes that bound the bins: fewer than 9, 9-12, 12-15, 15 or more.
STOP_BINS = (9.0, 12.0, 15.0)
N_BINS = len(STOP_BINS) + 1
# Weights of the label readings fixed by the owner for task 15: two thirds `controlled`. The IDSSE
# readings stand in for them by density (controlled1.5 like touch, controlled3.5 like controlled).
READING_WEIGHTS = {
    "touch": 1 / 3,
    "controlled": 2 / 3,
    "controlled1.5": 1 / 3,
    "controlled3.5": 2 / 3,
    "pff": 1.0,
}


def _codes(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype.kind in "UO":
        return pm.label_codes(labels)
    return labels.astype(np.int8)


def window_descriptors(labels, half: np.ndarray, fps: int) -> list[dict]:
    """Per window of `validation.windows(half, fps)`: stoppages per 10 minutes (stoppages wholly
    inside the window), the share of possessions that start right after a stoppage, possessions per
    live minute, and the window's stoppage-density bin (`STOP_BINS`)."""
    codes = _codes(labels)
    out = []
    for start, stop in validation.windows(half, fps):
        segs = pm.segments(codes, [(start, stop)])
        minutes = (stop - start) / fps / 60.0
        live = sum(b - a for a, b, k in segs if k in (pm.H, pm.A)) / fps / 60.0
        stoppages = sum(1 for a, b, k in segs if k == pm.D and a > start and b < stop)
        after = [
            segs[i - 1][2] == pm.D
            for i in range(1, len(segs))
            if segs[i][2] in (pm.H, pm.A) and segs[i - 1][1] == segs[i][0]
        ]
        density = stoppages / minutes * 10.0 if minutes > 0 else 0.0
        out.append(
            {
                "stoppages_per_10min": round(density, 4),
                "after_stoppage": round(float(np.mean(after)), 4) if after else None,
                "possessions_per_live_minute": (
                    round(sum(k in (pm.H, pm.A) for _, _, k in segs) / live, 4) if live else None
                ),
                "bin": int(np.digitize(density, STOP_BINS)),
            }
        )
    return out


def bin_shares(windows: list[dict]) -> np.ndarray:
    """(N_BINS,) share of the windows in each stoppage-density bin."""
    counts = np.bincount([w["bin"] for w in windows], minlength=N_BINS).astype(np.float64)
    return counts / counts.sum() if counts.sum() else counts


def importance(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """(N_BINS,) weight of a window in each bin: the target's share over the source's share; 0 where
    the source has no window (that part of the target is not covered and is reported, not
    extrapolated)."""
    target, source = np.asarray(target, float), np.asarray(source, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(source > 0, target / source, 0.0)


def coverage(target: np.ndarray, source: np.ndarray) -> float:
    """Share of the target's windows that fall in bins the source has windows in."""
    return float(np.sum(np.asarray(target)[np.asarray(source) > 0]))


def weighted_mean(values, weights) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    total = weights.sum()
    return float(np.sum(values * weights) / total) if total > 0 else float("nan")


def weighted_bootstrap(values, weights, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """5-95% interval of the weighted mean of `values` when the (value, weight) blocks are resampled
    with replacement."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(draws, len(values)))
    w = weights[picks]
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.sum(values[picks] * w, axis=1) / np.sum(w, axis=1)
    means = means[np.isfinite(means)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))
