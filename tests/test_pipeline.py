import numpy as np
import pytest

import possession_metrics as pm
from possession import config, pipeline, scoring
from possession._paths import CONFIGS_DIR

BASE = {
    "name": "t",
    "ball_state": {"method": "ball_visible"},
    "team": {"method": "nearest_player"},
    "decode": {"method": "majority_vote", "window": 10},
}


def test_every_committed_config_parses():
    paths = sorted(CONFIGS_DIR.glob("*.toml"))
    assert paths
    for path in paths:
        config.load(path)


def test_unknown_parameter_is_an_error():
    bad = {**BASE, "decode": {"method": "majority_vote", "windw": 10}}
    with pytest.raises(ValueError, match="no parameter"):
        config.parse(bad)


def test_unknown_method_is_an_error():
    with pytest.raises(KeyError, match="registered"):
        config.parse({**BASE, "team": {"method": "nope"}})


def test_run_produces_a_valid_submission(match, tmp_path):
    result = pipeline.run(config.parse(BASE), match)
    assert scoring.validate(result.labels, match.n_frames) is None
    assert (result.labels[~match.graded] == "X").all()

    sha = pipeline.write_csv(tmp_path / "t.csv", result.labels)
    table = pm.read_label_csv(tmp_path / "t.csv")
    assert pm.validate_submission(table, match.n_frames) is None
    assert sha == pipeline.write_csv(tmp_path / "again.csv", result.labels)


def test_diagnostics(match):
    result = pipeline.run(config.parse(BASE), match)
    diag = scoring.diagnostics(result.labels, match.half, match.fps)
    assert diag["n_possessions"] > 0
    assert abs(sum(diag["label_share_graded"].values()) - 1) < 1e-6


def test_pipeline_rejects_bad_stage_output(match, monkeypatch):
    import dataclasses

    from possession import registry

    broken = dataclasses.replace(
        registry.get("ball_state", "ball_visible"), fn=lambda m: np.full(m.n_frames, 2.0)
    )
    monkeypatch.setitem(registry._REGISTRY["ball_state"], "ball_visible", broken)
    with pytest.raises(ValueError, match="outside"):
        pipeline.run(config.parse(BASE), match)
