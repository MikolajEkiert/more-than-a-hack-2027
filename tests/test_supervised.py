"""The bridge to the supervised package solution/: the match layout it takes, the label functions
it can be trained on, and the two stage methods. The model itself is replaced by a stub here
(tests/conftest.py); training it needs the match data and the open data."""

import gzip
import hashlib

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

    def probabilities(m, labels="raw", data="idsse", regime="none", tta=False):
        seen.append((labels, data, regime, tta))
        return p

    monkeypatch.setattr(supervised, "probabilities", probabilities)
    alive = supervised_ball_state(match)
    home = supervised_team(match, labels="tau2", data="pff")
    supervised_ball_state(match, regime="auto_lf", tta=True)

    np.testing.assert_allclose(alive, 1.0 - p[:, 2])
    assert np.isnan(home[5]) and alive[5] == 0.0
    known = ~np.isnan(home)
    np.testing.assert_allclose(home[known], p[known, 0] / (p[known, 0] + p[known, 1]))
    # every parameter reaches the model
    assert seen == [
        ("raw", "idsse", "none", False),
        ("tau2", "pff", "none", False),
        ("raw", "idsse", "auto_lf", True),
    ]


def test_hybrid_config_runs_through_the_pipeline():
    match = make_match()
    result = pipeline.run(config_mod.load(CONFIGS_DIR / "hybrid_supervised.toml"), match)
    assert set(np.unique(result.labels)) <= {"H", "A", "D", "X"}
    assert (result.labels[~match.graded] == "X").all()
    assert {"H", "A", "D"} & set(np.unique(result.labels[match.graded]))


def test_committed_models_are_unpacked_into_the_package_cache(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    text = b"tree\nversion=v4\n"
    (models / "abc.txt.gz").write_bytes(gzip.compress(text, mtime=0))
    (models / "sha256").write_text(f"{hashlib.sha256(text).hexdigest()}  abc.txt\n")
    cache = tmp_path / "cache"

    assert supervised.seed_model_cache(models, cache) == ["abc.txt"]
    assert (cache / "models" / "abc.txt").read_bytes() == text
    assert supervised.seed_model_cache(models, cache) == []  # already there: nothing to do

    (cache / "models" / "abc.txt").write_bytes(b"local")  # a cache entry is never overwritten
    supervised.seed_model_cache(models, cache)
    assert (cache / "models" / "abc.txt").read_bytes() == b"local"

    (models / "sha256").write_text("0" * 64 + "  abc.txt\n")  # a file that is not the listed one
    with pytest.raises(ValueError):
        supervised.seed_model_cache(models, tmp_path / "other")
    assert not (tmp_path / "other" / "models" / "abc.txt").exists()


def test_models_in_the_repository_match_their_listing():
    lines = (supervised.MODELS_DIR / "sha256").read_text().splitlines()
    listing = dict(line.split()[::-1] for line in lines if line)
    packed = sorted(supervised.MODELS_DIR.glob("*.txt.gz"))
    assert packed and {p.name[: -len(".gz")] for p in packed} == set(listing)
    for path in packed:
        text = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(text).hexdigest() == listing[path.name[: -len(".gz")]]
        assert text.startswith(b"tree\n")  # a LightGBM text model


def test_a_container_round_trips_through_a_match():
    match = make_match(n=400)
    back = supervised.to_match(supervised.container(match))
    for key in ("px", "py", "bx", "by", "team", "half"):
        np.testing.assert_array_equal(
            supervised.container(back)[key], supervised.container(match)[key]
        )
    assert back.fps == 25 and back.game == match.game
    assert back.players.shape == match.players.shape
    assert np.isnan(back.ball[:, 2]).all()  # the container carries no ball speed
    moving = ~np.isnan(back.players[:, :, 2])
    assert moving.any() and (back.players[:, :, 2][moving] >= 0).all()  # speed from positions


@pytest.mark.needs_model  # keeps the real `probabilities`; the model itself is never built here
def test_a_training_match_is_only_ever_scored_out_of_fold(monkeypatch):
    from possession import validation

    def no_model(labels="raw", data="idsse", regime="none"):
        raise AssertionError("the full model must not be used for a training match")

    monkeypatch.setattr(supervised, "model", no_model)
    base = make_match(n=300)
    held = supervised.to_match({**supervised.container(base), "name": "IDSSE-TEST_deg0"})
    asked = []

    def out_of_fold(labels="raw", data="idsse", regime="none", tta=False):
        asked.append((labels, data, regime, tta))
        return fake_probabilities(held)

    held._cache[supervised.HELD_OUT] = out_of_fold
    np.testing.assert_array_equal(supervised.probabilities(held), fake_probabilities(held))
    supervised.probabilities(held, "tau2", "idsse+pff")
    supervised.probabilities(held, regime="auto_ml", tta=True)
    # one out-of-fold prediction per model variant, never the full model
    assert asked == [
        ("raw", "idsse", "none", False),
        ("tau2", "idsse+pff", "none", False),
        ("raw", "idsse", "auto_ml", True),
    ]

    bare = supervised.to_match({**supervised.container(base), "name": "IDSSE-TEST_deg0"})
    with pytest.raises(RuntimeError):
        supervised.probabilities(bare)  # a training match without an out-of-fold prediction
    truth = np.where(base.graded, "H", "X")
    degraded = validation.degrade(held, truth, seed=0)  # a copy loses the attached prediction
    with pytest.raises(RuntimeError):
        supervised.probabilities(degraded)


def test_pff_matches_are_split_once_and_training_data_is_named(monkeypatch, tmp_path):
    from solution import pff

    for game_id in range(3812, 3828):  # sixteen converted matches
        (tmp_path / f"{game_id}.npz").write_bytes(b"")
    monkeypatch.setattr(pff, "NPZ_DIR", str(tmp_path))
    supervised.pff_games.cache_clear()
    try:
        train, test = supervised.pff_games()
    finally:
        supervised.pff_games.cache_clear()
    assert test == ("3815", "3819", "3823", "3827")  # sorted by id, every fourth
    assert len(train) == 12 and not set(train) & set(test)

    with pytest.raises(ValueError):
        supervised.model("raw", "everything")  # training data is one of supervised.DATA
