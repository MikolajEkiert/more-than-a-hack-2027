"""Validation, label-free diagnostics and the public-window sanity check.

Everything here delegates to evaluation/possession_metrics.py, the organisers' specification.
Nothing is reimplemented, so a number printed here is the number the official scorer would print.

The public 5-minute window is a SANITY CHECK, not a model-selection signal (RULES.md 2,
EVALUATION.md "The public window is not a leaderboard"). Its result is reported under the key
`public_sanity` so that nobody mistakes it for a validation score.
"""

import numpy as np
import pandas as pd

import possession_metrics as pm
from possession._paths import GT_DIR


def validate(labels: np.ndarray, n_expected: int) -> str | None:
    """None if `labels` would pass the official validator, else the one-line reason."""
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    return pm.validate_submission(table, n_expected)


def diagnostics(labels: np.ndarray, half: np.ndarray, fps: int) -> dict:
    """What the submission claims, computed with no ground truth.

    Possession counts and durations are the earliest warning of flicker (too many, too short) or
    over-smoothing (too few, too long). Compare them with football knowledge, not with the GT.
    """
    codes = pm.label_codes(labels)
    graded = half > 0
    runs = _runs(graded)
    filled = pm.fill_abstentions(codes, runs)
    stats = pm.possession_stats(filled, runs)
    n_graded = int(graded.sum())
    share = {
        name: float(np.sum(codes[graded] == index)) / n_graded if n_graded else 0.0
        for index, name in enumerate(pm.LABELS)
    }
    return {
        "n_possessions": int(stats["n_spells"]),
        "mean_possession_s": round(stats["mean_duration_s"], 2),
        "home_share": round(stats["home_share"], 4),
        "label_share_graded": {k: round(v, 4) for k, v in share.items()},
        "n_frames": int(len(codes)),
        "fps": fps,
    }


def public_sanity(labels: np.ndarray, game: str) -> dict | None:
    """Scores against the public 5-minute window, or None if this game has none."""
    path = GT_DIR / f"{game}_possession_gt_public.csv"
    if not path.is_file():
        return None
    gt = pm.label_codes(pm.read_label_csv(path))
    if len(gt) != len(labels):
        return None
    result = pm.evaluate(gt, pm.label_codes(labels))
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "n_pred_possessions": result["n_pred_spells"],
        "n_gt_possessions": result["n_gt_spells"],
        "note": "public 5 min, 17 possessions: sanity only, never for choosing between versions",
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )
