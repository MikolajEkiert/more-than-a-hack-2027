"""Repository paths, and the one place that makes the organiser modules importable.

`games`, `evaluation/possession_metrics.py` and `starter/` live at the repository root and are
owned by the organisers. We import them instead of copying them, so the root goes on sys.path.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
GT_DIR = EVALUATION_DIR / "gt"
CONFIGS_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
SUBMISSION_DIR = REPO_ROOT / "submission"
CACHE_DIR = Path(os.environ.get("POSSESSION_CACHE_DIR") or REPO_ROOT / ".cache")

for _path in (str(EVALUATION_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
