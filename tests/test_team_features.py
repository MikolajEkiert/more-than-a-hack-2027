"""The team feature block (possession/team_features.py) and the feature set "v1+team" of the
supervised model (possession/supervised.py), on synthetic containers only."""

import numpy as np
import pytest

from possession import supervised
from possession import team_features as tf
from tests.conftest import make_match

HALF = 1


def _container(px, py, team, bx, by, half=None):
    """A package container from per-frame arrays (frames x slots)."""
    px, py = np.asarray(px, np.float32), np.asarray(py, np.float32)
    n = len(px)
    return {
        "px": px,
        "py": py,
        "team": np.asarray(team, np.int8),
        "bx": np.asarray(bx, np.float32),
        "by": np.asarray(by, np.float32),
        "half": np.full(n, HALF, np.int8) if half is None else np.asarray(half, np.int8),
        "pitch": np.array([105.0, 68.0], np.float32),
        "label": np.full(n, 3, np.int8),
        "name": "SYNTH",
    }


def _column(block, names, name):
    return block[:, names.index(name)].astype(np.float64)


def test_block_layout_on_a_synthetic_match():
    m = supervised.container(make_match(n=1200))
    block, names = tf.build(m)
    assert block.dtype == np.float16 and block.shape == (1200, len(names))
    assert len(set(names)) == len(names) and names == tf.feature_names()
    assert all(name.startswith("t_") for name in names)  # never a name of the package's set
    for key in tf.CONTEXT_KEYS:  # every main cue gets its past, future and centred means
        assert {f"{key}_{t}" for t in ("c25", "p25", "p100", "f25", "f100")} <= set(names)
    # cues of the tracked ball are undefined where it is missing
    seen = ~np.isnan(m["bx"])
    for name in ("t_gs_diff", "t_gsl_n_h", "t_npress_a"):
        column = _column(block, names, name)
        assert np.isnan(column[~seen]).all() and not np.isnan(column[seen]).all()
    np.testing.assert_array_equal(tf.build(m)[0], block)  # deterministic


def test_orientation_is_the_packages():
    _, _, _, package_features = supervised._package()
    match = make_match(n=600)
    m = supervised.container(match)
    m["half"][300:] = 2
    m["px"][300:] = 105.0 - m["px"][300:]  # the second half is played the other way round
    for ours, theirs in zip(tf.orient(m), package_features.orient(m), strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_goal_side_counts_players_between_ball_and_own_goal():
    # home (slots 0-2) defends x = 0, away (slots 3-5) defends x = 105: home's mean x is lower
    px = np.tile([10.0, 40.0, 60.0, 55.0, 70.0, 30.0], (60, 1))
    py = np.tile([34.0, 34.0, 34.0, 34.0, 60.0, 34.0], (60, 1))
    bx, by = np.full(60, 50.0), np.full(60, 34.0)
    bx[20:40] = np.nan  # a gap: the last and next positions still read the arrangement
    block, names = tf.build(_container(px, py, [0, 0, 0, 1, 1, 1], bx, by))
    frame = 10
    assert _column(block, names, "t_gs_n_h")[frame] == 2  # x = 10 and 40 are behind the ball
    assert _column(block, names, "t_gs_n_a")[frame] == 2  # x = 55 and 70
    assert _column(block, names, "t_gsl_n_a")[frame] == 1  # y = 60 is 26 m across: not near
    np.testing.assert_allclose(_column(block, names, "t_gs_share_h")[frame], 2 / 3, atol=1e-3)
    assert np.isnan(_column(block, names, "t_gs_diff")[30])
    assert _column(block, names, "t_gs_diff_last")[30] == pytest.approx(0.0, abs=1e-3)
    assert _column(block, names, "t_gs_diff_next")[30] == pytest.approx(0.0, abs=1e-3)


def test_association_tells_the_possessor_from_a_player_passing_by():
    n = 100
    t = np.arange(n, dtype=np.float64)
    ball_x = 40.0 + 0.2 * t  # carried at 5 m/s
    carrier = ball_x - 1.2  # home player 1.2 m behind the ball, all the way
    runner = 70.0 - 0.6 * t  # an away player crossing the ball's path at 15 m/s
    px = np.column_stack([carrier, np.full(n, 10.0), runner, np.full(n, 95.0)])
    py = np.column_stack([np.full(n, 34.0), np.full(n, 30.0), np.full(n, 34.3), np.full(n, 30.0)])
    block, names = tf.build(_container(px, py, [0, 0, 1, 1], ball_x, np.full(n, 34.0)))
    crossing = int(np.argmin(np.abs(runner - ball_x)))
    assert np.abs(runner - ball_x)[crossing] < 1.2  # at that frame the away player is nearer
    assert _column(block, names, "t_assoc_diff_30")[crossing] < -3.0  # over +-1.2 s, home is
    assert _column(block, names, "t_assoc_h_5")[crossing] == pytest.approx(1.2, abs=0.05)
    rv_h, rv_a = _column(block, names, "t_rv_h"), _column(block, names, "t_rv_a")
    assert rv_h[crossing] < 1.0 < 10.0 < rv_a[crossing]  # one moves with the ball, one does not


def test_swapped_view_exchanges_the_teams():
    """With home and away exchanged, every per-team cue moves to the other team and every
    difference changes sign (the second orientation the model is trained on). Frames outside
    the halves are not oriented, as in the package, and never train the model: they are left
    out, with the widest context window (100 frames) and the carry cues' reach (velocity over
    +-2, mean over +-5 frames) after them."""
    m = supervised.container(make_match(n=900, seed=3, ungraded_head=200))
    inside = slice(310, None)
    block, names = tf.build(m)
    swapped, names_s = tf.build(tf.swapped(m))
    assert names_s == names
    block, swapped = block[inside], swapped[inside]

    def partner(name):
        for a, b in (("_h", "_a"), ("_a", "_h")):
            for tail in ("", "_5", "_15", "_30"):
                if name.endswith(a + tail) and "_diff" not in name:
                    return name[: len(name) - len(a + tail)] + b + tail, 1.0
        return name, -1.0 if "_diff" in name else 1.0

    for j, name in enumerate(names):
        other, sign = partner(name)
        np.testing.assert_allclose(
            swapped[:, j].astype(np.float64),
            sign * block[:, names.index(other)].astype(np.float64),
            atol=0.02,
            rtol=0.01,
            err_msg=name,
        )


def test_rolling_mean_is_nan_aware():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    np.testing.assert_allclose(tf.rolling_mean(x, -1, 1), [1.0, 2.0, 3.0, 3.0, 6.0, 6.0])
    out = tf.rolling_mean(x, -1, 1, min_count=2)
    np.testing.assert_array_equal(np.isnan(out), [True, False, True, True, True, True])
    two = tf.rolling_mean(np.column_stack([x, 2 * x]), 0, 0)
    np.testing.assert_array_equal(two[:, 1], 2 * x)


def test_block_cache_gives_the_computed_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(supervised, "TEAM_DIR", tmp_path)
    m = supervised.container(make_match(n=500))
    first = supervised.team_block(m)
    assert len(list((tmp_path / "blocks").glob("*.npy"))) == 1
    np.testing.assert_array_equal(np.asarray(supervised.team_block(m)), np.asarray(first))
    np.testing.assert_array_equal(np.asarray(first), tf.build(m)[0])
    np.testing.assert_array_equal(
        np.asarray(supervised.team_block(m, "swap")), tf.build(tf.swapped(m))[0]
    )
    monkeypatch.setenv("POSSESSION_NO_FEATURE_CACHE", "1")
    np.testing.assert_array_equal(supervised.team_block(m), np.asarray(first))
    with pytest.raises(ValueError):
        supervised.team_block(m, "mirror")


def test_training_rows_are_the_packages_rows_with_the_block_appended():
    from solution import evaluate_external as ev

    rng = np.random.default_rng(0)
    prepared, blocks = [], []
    for j, n in enumerate((40, 55, 33)):
        label = rng.integers(0, 4, n).astype(np.int8)  # codes H A D X: X rows are never used
        swap_label = np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)
        prepared.append(
            {
                "m": {"label": label},
                "X": rng.random((n, 3)).astype(np.float16),
                "Xs": rng.random((n, 3)).astype(np.float16),
                "ys": swap_label,
            }
        )
        blocks.append(
            {"orig": np.full((n, 2), j, np.float16), "swap": -np.ones((n, 2), np.float16)}
        )
    for skip in (None, 1):
        x, y = supervised._team_stack(prepared, blocks, skip=skip)
        x_ref, y_ref = ev.stack(prepared, skip=skip)
        np.testing.assert_array_equal(x[:, :3], x_ref)
        np.testing.assert_array_equal(y, y_ref)
        assert x.dtype == np.float16 and x.shape[1] == 5
        weights = ev.stack_weights(prepared, skip, ev.TRAIN_STRIDE, dict(ev.DEFAULT_CFG))
        assert len(weights) == len(y)
    # the block of the swapped orientation sits on the swapped rows
    x, _ = supervised._team_stack(prepared[:1], blocks[:1])
    half = len(x) // 2
    assert (x[:half, 3:] == 0).all() and (x[half:, 3:] == -1).all()


def test_unknown_feature_sets_and_options_are_errors():
    with pytest.raises(ValueError):
        supervised._check_team("v2")
    for options in ({"data": "pff"}, {"regime": "mix_lf"}, {"tta": True}):
        with pytest.raises(ValueError):
            supervised._check_team("v1+team", **options)
    supervised._check_team("v1+team")
    supervised._check_team("v1", data="pff", regime="mix_lf", tta=True)  # v1: not this check's


class _Booster:
    """Stands in for LightGBM: records the matrix and returns fixed probabilities."""

    def __init__(self):
        self.seen = []

    def predict(self, x):
        self.seen.append(np.asarray(x))
        return np.tile([0.5, 0.3, 0.2], (len(x), 1))


def _fake_v1(m):
    return np.column_stack([m["bx"], m["by"]]).astype(np.float32), ["bx", "by"]


@pytest.mark.needs_model  # keeps the real `probabilities`; the model is a stand-in
def test_the_default_path_asks_the_packages_model(monkeypatch):
    monkeypatch.setenv("POSSESSION_NO_FEATURE_CACHE", "1")
    booster, asked = _Booster(), []

    def fake_model(*args, **kwargs):
        asked.append((args, kwargs))
        return booster, _fake_v1

    monkeypatch.setattr(supervised, "model", fake_model)
    monkeypatch.setattr(supervised, "_predict_team", lambda *a: pytest.fail("not the default"))
    match = make_match(n=300)
    supervised.probabilities(match)
    supervised.probabilities(match, "tau2", "idsse+pff", regime="none")
    # the package's model, asked exactly as before the feature sets existed
    assert asked == [(("raw", "idsse", "none"), {}), (("tau2", "idsse+pff", "none"), {})]
    assert all(x.shape[1] == 2 for x in booster.seen)  # the package's features alone


@pytest.mark.needs_model
def test_the_team_set_appends_the_block(monkeypatch):
    monkeypatch.setenv("POSSESSION_NO_FEATURE_CACHE", "1")
    booster, asked = _Booster(), []

    def fake_model(*args, **kwargs):
        asked.append(args)
        return booster, _fake_v1

    monkeypatch.setattr(supervised, "model", fake_model)
    match = make_match(n=300)
    p = supervised.probabilities(match, features="v1+team")
    assert p.shape == (300, 3) and asked == [("raw", "idsse", "none", "v1+team")]
    (x,) = booster.seen
    block = tf.build(supervised.container(match))[0]
    assert x.shape == (300, 2 + block.shape[1])
    np.testing.assert_array_equal(x[:, 2:], block)
    v1 = _fake_v1(supervised.container(match))[0].astype(np.float16)  # the package's dtype
    np.testing.assert_array_equal(x[:, :2], v1)
    with pytest.raises(ValueError):
        supervised.probabilities(make_match(n=300), features="v1+team", tta=True)
