"""The bridge to the supervised package solution/: the match layout it takes, the label functions
it can be trained on, and the two stage methods. The model itself is replaced by a stub here
(tests/conftest.py); training it needs the match data and the open data."""

import numpy as np
import pytest

import possession_metrics as pm
from possession import config as config_mod
from possession import pipeline, supervised
from possession._paths import CONFIGS_DIR
from possession.stages.ball_state import supervised as supervised_ball_state
from possession.stages.team import supervised as supervised_team
from tests.conftest import fake_probabilities, make_match


def test_container_has_the_layout_of_the_package():
    match = make_match(n=500)
    before = (match.ball.copy(), match.players.copy())
    m = supervised.container(match)
    n, slots = match.n_frames, match.players.shape[1]

    assert m["px"].shape == m["py"].shape == (n, slots) and m["px"].dtype == np.float32
    assert m["bx"].shape == m["by"].shape == (n,) and m["bx"].dtype == np.float32
    assert m["team"].dtype == np.int8 and m["half"].dtype == np.int8 and m["label"].dtype == np.int8
    assert (m["label"] == pm.X).all()  # the match is never a source of labels
    assert m["name"] == match.game
    np.testing.assert_allclose(m["pitch"], match.pitch, rtol=1e-6)
    np.testing.assert_array_equal(np.isnan(m["bx"]), ~match.ball_seen)  # unseen stays NaN
    np.testing.assert_allclose(m["px"], match.players[:, :, 0], rtol=1e-6, equal_nan=True)

    m["bx"][:] = 0.0  # the container owns its arrays: writing to it never reaches the match
    np.testing.assert_array_equal(match.ball, before[0])
    np.testing.assert_array_equal(match.players, before[1])


def test_label_functions_by_name():
    h, a, d = pm.H, pm.A, pm.D
    # 2 s home, 0.4 s away, 2 s home, a stoppage, 0.8 s away after the restart
    codes = np.array([h] * 50 + [a] * 10 + [h] * 50 + [d] * 20 + [a] * 20, dtype=np.int8)
    m = {"label": codes}

    np.testing.assert_array_equal(supervised.label_function("raw")(m), codes)
    merged = supervised.label_function("tau1")(m)
    assert (merged[:110] == h).all()  # the 0.4 s touch between two home spells is not a possession
    np.testing.assert_array_equal(merged[110:], codes[110:])  # nothing else moves
    np.testing.assert_array_equal(supervised.label_function("tau0.2")(m), codes)  # 0.4 s >= 0.2 s
    controlled = supervised.label_function("controlled2")(m)
    assert (controlled[:110] == h).all() and (controlled[130:] == a).all()  # restarts always stay

    for bad in ("", "tau", "taux", "smooth2"):
        with pytest.raises(ValueError):
            supervised.label_function(bad)


def test_stage_methods_read_the_three_class_probabilities(monkeypatch):
    match = make_match(n=600)
    p = fake_probabilities(match)
    p[5] = [0.0, 0.0, 1.0]  # certainly dead: the model says nothing about the team
    seen = []

    def probabilities(m, labels="raw"):
        seen.append(labels)
        return p

    monkeypatch.setattr(supervised, "probabilities", probabilities)
    alive = supervised_ball_state(match)
    home = supervised_team(match, labels="tau2")

    np.testing.assert_allclose(alive, 1.0 - p[:, 2])
    assert np.isnan(home[5]) and alive[5] == 0.0
    known = ~np.isnan(home)
    np.testing.assert_allclose(home[known], p[known, 0] / (p[known, 0] + p[known, 1]))
    assert seen == ["raw", "tau2"]  # `labels` reaches the model


def test_hybrid_config_runs_through_the_pipeline():
    match = make_match()
    result = pipeline.run(config_mod.load(CONFIGS_DIR / "hybrid_supervised.toml"), match)
    assert set(np.unique(result.labels)) <= {"H", "A", "D", "X"}
    assert (result.labels[~match.graded] == "X").all()
    assert {"H", "A", "D"} & set(np.unique(result.labels[match.graded]))
