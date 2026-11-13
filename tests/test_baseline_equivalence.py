"""The regression anchor: the pipeline, configured as the baseline, reproduces the organisers'
committed baseline_smooth50.csv frame for frame on every graded frame.

If this fails after a change to the loader, features or registered baseline methods, something
upstream of every experiment broke. Runs only with the official BAR-ATH data present.
"""

import numpy as np
import pytest

import games
import possession_metrics as pm
from possession import config, pipeline
from possession._paths import CONFIGS_DIR, RESULTS_DIR


def _have_data():
    try:
        games.require(["atd_meta", "atd_raw"], "BAR-ATH")
        return True
    except OSError:
        return False


@pytest.mark.needs_data
@pytest.mark.skipif(not _have_data(), reason="official BAR-ATH data not in data/")
def test_pipeline_reproduces_baseline_smooth50():
    from possession.loader import load_match

    match = load_match("BAR-ATH", verbose=False)
    result = pipeline.run(config.load(CONFIGS_DIR / "baseline_smooth50.toml"), match)
    expected = pm.read_label_csv(RESULTS_DIR / "baseline_smooth50.csv")["label"].to_numpy()
    graded = match.graded
    mismatches = np.flatnonzero(result.labels[graded] != expected[graded].astype("<U1"))
    assert mismatches.size == 0, f"{mismatches.size} graded frames differ"
