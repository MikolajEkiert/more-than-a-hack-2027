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
    for options in ({"data": "pff"}, {"regime": "mix_lf"}):
        with pytest.raises(ValueError):
            supervised._check_team("v1+team", **options)
    supervised._check_team("v1+team")
    supervised._check_team("v1+team", tta=True)  # task 18: swap-averaging exists for the block
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


class _SideBooster:
    """Says P(H) = 0.7 when the first block column is the home view's, 0.1 otherwise."""

    def __init__(self, width, column):
        self.width, self.column, self.seen = width, column, []

    def predict(self, x):
        x = np.asarray(x)
        self.seen.append(x)
        home_view = np.isclose(x[:, self.width], self.column, equal_nan=True).all()
        row = [0.7, 0.2, 0.1] if home_view else [0.1, 0.8, 0.1]
        return np.tile(row, (len(x), 1))


@pytest.mark.needs_model
def test_team_set_tta_averages_the_swapped_view(monkeypatch):
    """Task 18: with tta, the swapped match is predicted too (package features of the swapped
    container, block of the swapped view) and its P(H) / P(A) are exchanged back before the mean."""
    monkeypatch.setenv("POSSESSION_NO_FEATURE_CACHE", "1")
    match = make_match(n=300)
    m = supervised.container(match)
    block = tf.build(m)[0]
    swapped_block = tf.build(tf.swapped(m))[0]
    j = int(np.argmax(np.any(block != swapped_block, axis=0)))  # a column the views differ on
    assert not np.array_equal(block[:, j], swapped_block[:, j], equal_nan=True)
    booster = _SideBooster(2, block[:, j])

    def fake_v1(c):
        return np.column_stack([c["bx"], c["by"]]).astype(np.float32), ["bx", "by"]

    monkeypatch.setattr(supervised, "model", lambda *a: (booster, fake_v1))
    plain = supervised.probabilities(match, features="v1+team-tw2")
    np.testing.assert_allclose(plain, np.tile([0.7, 0.2, 0.1], (300, 1)))
    p = supervised.probabilities(match, features="v1+team-tw2", tta=True)
    # orig view: (0.7, 0.2, 0.1); swapped view (0.1, 0.8, 0.1) read back as (0.8, 0.1, 0.1)
    np.testing.assert_allclose(p, np.tile([0.75, 0.15, 0.1], (300, 1)))
    assert len(booster.seen) == 3  # the plain call, then the two views of the tta call
    np.testing.assert_array_equal(booster.seen[2][:, 2:], swapped_block)
    from solution import container as package_container

    swapped_v1 = fake_v1(package_container.swap_teams(m))[0].astype(np.float16)
    np.testing.assert_array_equal(booster.seen[2][:, :2], swapped_v1)


def test_turn_multiplier_covers_team_changes_only():
    """Task 18: frames within `frames` of an H/A hand-over get the factor; stoppage edges, X and
    a change across a stoppage do not."""
    from solution import container as C

    h, a, d, x = C.H, C.A, C.D, C.X
    label = np.array([h] * 6 + [a] * 6 + [d] * 4 + [h] * 5 + [x] * 2 + [a] * 3, np.int8)
    w = supervised.turn_multiplier(label, frames=2, factor=4.0)
    expected = np.ones(len(label))
    expected[4:9] = 4.0  # the change at frame 6, +-2 frames
    np.testing.assert_array_equal(w, expected)
    np.testing.assert_array_equal(supervised.turn_multiplier(label, 2, 1.0), np.ones(len(label)))
    # clipped at the ends, overlapping windows merge
    short = np.array([h, a, h, h, h, h], np.int8)
    np.testing.assert_array_equal(
        supervised.turn_multiplier(short, 1, 2.0), [2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    )
    assert supervised.turn_multiplier(np.array([h], np.int8), 3, 2.0).tolist() == [1.0]


def test_turn_rows_align_with_the_training_rows():
    """The multiplier sits on the rows `_team_stack` builds (stride, X rows dropped, the swapped
    orientation after the original), and the turn-weighted sets carry their options in the key."""
    from solution import container as C
    from solution import evaluate_external as ev

    rng = np.random.default_rng(1)
    prepared = []
    for n in (60, 47):
        label = rng.choice([C.H, C.A, C.D, C.X], size=n, p=[0.45, 0.45, 0.05, 0.05])
        label = label.astype(np.int8)
        swap = np.where(label == C.H, C.A, np.where(label == C.A, C.H, label)).astype(np.int8)
        prepared.append({"m": {"label": label}, "X": None, "Xs": object(), "ys": swap})
    for skip in (None, 0):
        got = supervised._team_turn_rows(prepared, skip, frames=3, factor=2.0)
        want = []
        for j, d in enumerate(prepared):
            if j == skip:
                continue
            for y in (d["m"]["label"], d["ys"]):
                rows = np.flatnonzero(y != C.X)[:: ev.TRAIN_STRIDE]
                want.append(supervised.turn_multiplier(y, 3, 2.0)[rows])
        np.testing.assert_array_equal(got, np.concatenate(want))
        assert set(np.unique(got)) <= {1.0, 2.0}
    assert supervised._training_options("v1+team") == {}
    assert supervised._training_options("v1+team-r600") == {}
    assert supervised._training_options("v1+team-tw4") == {"turn_weight": 4.0, "turn_s": 1.0}


def test_committed_team_model_is_listed_and_unpacks(tmp_path):
    """Every committed team-feature model is listed with the sha256 of its text and unpacks into a
    cache folder under its own name (task 14: the submission must not depend on retraining)."""
    import gzip
    import hashlib

    from possession import supervised

    packed = sorted(supervised.TEAM_MODELS_DIR.glob("*.txt.gz"))
    assert packed, "no committed team-feature model"
    listing = dict(
        line.split()[::-1]
        for line in (supervised.TEAM_MODELS_DIR / "sha256").read_text().splitlines()
        if line
    )
    for path in packed:
        text = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(text).hexdigest() == listing[path.name[: -len(".gz")]]
    written = supervised.seed_team_models(cache_dir=tmp_path)
    assert sorted(written) == sorted(p.name[: -len(".gz")] for p in packed)
    assert all((tmp_path / "models" / name).is_file() for name in written)


# -- task 17: extra training matches of other providers --------------------------------------


def _labelled(seed: int, n: int = 1500) -> dict:
    """A synthetic container with a partly missing ball and three long spells, H, D, A."""
    m = supervised.container(make_match(n=n, seed=seed, ungraded_head=0))
    third = n // 3
    m["label"] = np.repeat(np.array([0, 2, 1], np.int8), [third, third, n - 2 * third])
    m["name"] = f"EXTRA-{seed}"
    return m


def _package_setup(monkeypatch):
    """`supervised._setup` without the competition match: fixed degradation parameters."""
    from solution import features as package_features

    feature_fn = package_features.FEATURE_SETS["v1"]
    params = {"ball_missing": 0.4}
    monkeypatch.setattr(supervised, "_setup", lambda: (params, feature_fn, {"weights": True}))
    return params, feature_fn


def test_extra_rows_are_the_packages_extra_rows_with_the_block_appended(monkeypatch):
    """`_extra_match_rows` (one match at a time) gives what the package's `prepare` + `stack` /
    `stack_weights` make of the same list of extra matches, as `train_final` adds them."""
    from solution import cache as package_cache
    from solution import evaluate_external as ev
    from solution import labels as package_labels

    monkeypatch.setattr(package_cache, "ENABLED", False)
    monkeypatch.setenv("MTAH_NO_CACHE", "1")
    monkeypatch.setenv("POSSESSION_NO_FEATURE_CACHE", "1")
    params, feature_fn = _package_setup(monkeypatch)
    cfg = dict(ev.DEFAULT_CFG, weights=True)
    raws = [_labelled(seed) for seed in (1, 2)]
    prepared = ev.prepare(
        raws, feature_fn, package_labels.raw, seed=0, degrade_params=params,
        log=lambda text: None, swap=False,
    )  # fmt: skip
    x_ref, y_ref = ev.stack(prepared, stride=supervised.EXTRA_STRIDE)
    w_ref = ev.stack_weights(prepared, stride=supervised.EXTRA_STRIDE, cfg=cfg)
    parts = [
        supervised._extra_match_rows(raw, i, package_labels.raw, cfg) for i, raw in enumerate(raws)
    ]
    x = np.concatenate([p[0] for p in parts])
    n_package = x_ref.shape[1]
    np.testing.assert_array_equal(x[:, :n_package], x_ref)
    np.testing.assert_array_equal(np.concatenate([p[1] for p in parts]), y_ref)
    np.testing.assert_array_equal(np.concatenate([p[2] for p in parts]), w_ref)
    stride = supervised.EXTRA_STRIDE
    blocks = [
        supervised.team_block(d["m"])[np.flatnonzero(d["m"]["label"] != 3)[::stride]]
        for d in prepared
    ]
    np.testing.assert_array_equal(x[:, n_package:], np.concatenate(blocks))


def test_training_data_options_of_the_team_models():
    supervised._check_team("v1+team", data="idsse+pff")
    with pytest.raises(ValueError):
        supervised._check_team("v1+team", data="idsse+metrica")  # Metrica is never trained on
    assert supervised.TEAM_DATA["idsse"] == ()  # version 4's model: no extra matches


def test_version_4_keeps_its_model_key(monkeypatch):
    """The key of version 4's team model does not depend on the new option; another option
    gets its own key."""
    _package_setup(monkeypatch)
    seen = []
    monkeypatch.setattr(supervised, "_cached_booster", lambda path, fit: seen.append(path.name))
    monkeypatch.setattr(supervised, "seed_team_models", lambda: [])
    monkeypatch.setattr(supervised, "_extra_key", lambda data: ("extra", data))
    supervised._train_final_team.cache_clear()
    try:
        supervised._train_final_team("raw", "v1+team")
        supervised._train_final_team("raw", "v1+team", "idsse")
        supervised._train_final_team("raw", "v1+team", "idsse+pff")
    finally:
        supervised._train_final_team.cache_clear()
    assert seen[0] == seen[1] != seen[2]
