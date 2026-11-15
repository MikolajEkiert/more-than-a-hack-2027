"""Descriptors of possession structure per window and the weights built on them (task 15)."""

import numpy as np

from possession import composition

FPS = 25
MIN = 60 * FPS


def _labels(segments, n):
    out = np.full(n, "X", dtype="<U1")
    pos = 0
    for label, seconds in segments:
        frames = int(seconds * FPS)
        out[pos : pos + frames] = label
        pos += frames
    return out


def test_window_descriptors_count_stoppages_inside_each_window():
    n = 20 * MIN  # one half of 20 minutes: two 10-minute windows
    half = np.ones(n, dtype=np.int8)
    # window 1: H 100 s, D 10 s, A 100 s, D 10 s, H rest; window 2: one long H possession
    first = [("H", 100), ("D", 10), ("A", 100), ("D", 10), ("H", 380)]
    labels = _labels(first + [("H", 600)], n)
    out = composition.window_descriptors(labels, half, FPS)
    assert len(out) == 2
    assert out[0]["stoppages_per_10min"] == 2.0
    # the first H's predecessor lies outside the window, so only A and the second H count, and
    # both follow a stoppage
    assert out[0]["after_stoppage"] == 1.0
    assert out[0]["bin"] == 0 and out[1]["bin"] == 0
    assert out[1]["stoppages_per_10min"] == 0.0


def test_bins_follow_the_density_bounds():
    n = 10 * MIN
    half = np.ones(n, dtype=np.int8)
    segments = []
    for _ in range(13):  # 13 short stoppages inside one 10-minute window
        segments += [("H", 40), ("D", 4)]
    segments.append(("H", 600 - 13 * 44))
    out = composition.window_descriptors(_labels(segments, n), half, FPS)
    assert out[0]["stoppages_per_10min"] == 13.0
    assert out[0]["bin"] == 2  # 12-15


def test_importance_weights_and_coverage():
    target = np.array([0.1, 0.3, 0.4, 0.2])
    source = np.array([0.25, 0.5, 0.25, 0.0])
    weights = composition.importance(target, source)
    assert np.allclose(weights, [0.4, 0.6, 1.6, 0.0])
    assert composition.coverage(target, source) == 0.8
    # weighting the source's windows reproduces the target's shares where the source has windows
    assert np.allclose(source * weights, [0.1, 0.3, 0.4, 0.0])


def test_weighted_mean_and_bootstrap():
    values = np.array([0.0, 1.0, 2.0, 3.0])
    assert composition.weighted_mean(values, np.ones(4)) == 1.5
    assert composition.weighted_mean(values, [0, 0, 0, 1]) == 3.0
    low, high = composition.weighted_bootstrap(values, np.ones(4), draws=500)
    assert low < 1.5 < high
    again = composition.weighted_bootstrap(values, np.ones(4), draws=500)
    assert again == (low, high)  # seeded


def _grid14():
    import importlib.util

    from possession._paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location("grid14", REPO_ROOT / "scripts" / "grid14.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_comparison_with_unit_weights_is_the_plain_window_mean():
    grid14 = _grid14()
    a, b = [], []
    for game in ("g1", "g2"):
        for seed in (0, 1):
            for reading in ("touch", "controlled"):
                key = {"game": game, "seed": seed, "reading": reading}
                a.append({**key, "spellF1": 0.6, "windows": [0.5 + seed / 10, 0.7, None]})
                b.append({**key, "spellF1": 0.5, "windows": [0.4, 0.6 + seed / 10, 0.3]})
    unit = {"touch": 1.0, "controlled": 1.0}
    out = grid14.weighted(a, b, "label_free", window_weights={}, reading_weights=unit)
    diffs = [0.1 + s / 10 for s in (0, 1)] + [0.1 - s / 10 for s in (0, 1)]
    assert np.isclose(out["mean"], np.mean(diffs))  # windows 0 and 1, both games alike
    # two thirds on `controlled` changes nothing when both readings agree
    again = grid14.weighted(a, b, "label_free", window_weights={})
    assert np.isclose(again["mean"], out["mean"])
    # a weight of 0 on window 1 leaves window 0 only
    zero_one = {(g, 0): 1.0 for g in ("g1", "g2")} | {(g, 1): 0.0 for g in ("g1", "g2")}
    only0 = grid14.weighted(a, b, "label_free", window_weights=zero_one, reading_weights=unit)
    assert np.isclose(only0["mean"], 0.15)


def _composition15():
    import importlib.util

    from possession._paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "composition15", REPO_ROOT / "scripts" / "composition15.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stoppage_errors_separate_spurious_split_and_missed_stoppages():
    c15 = _composition15()
    n = 10 * MIN
    half = np.ones(n, dtype=np.int8)
    truth = _labels([("H", 100), ("D", 20), ("A", 100), ("D", 10), ("H", 370)], n)
    # a 2 s stoppage inside true play, the first true stoppage split by a 4 s live spell, the
    # second true stoppage missed
    pred = _labels(
        [("H", 50), ("D", 2), ("H", 48), ("D", 8), ("H", 4), ("D", 8), ("A", 110), ("H", 370)], n
    )
    out = c15.stoppage_errors(pred, truth, half, FPS)
    assert out["true"] == 2 and out["pred"] == 3
    assert out["spurious"] == 1 and out["split_extra"] == 1 and out["missed"] == 1
    assert out["spurious_lengths"] == [2.0]


def test_instrument_weights_bin_both_sides_by_the_same_labels(tmp_path, monkeypatch):
    import json

    c15 = _composition15()
    monkeypatch.setattr(c15, "OUT", tmp_path)
    (tmp_path / "pred" / "v").mkdir(parents=True)
    # two seeds of one game, two windows: mean densities 8 (bin 0) and 13 (bin 2)
    rows = [
        {"game": "g", "seed": seed, "windows": [{"stoppages_per_10min": d} for d in densities]}
        for seed, densities in ((0, (7.0, 12.0)), (1, (9.0, 14.0)))
    ]
    (tmp_path / "pred" / "v" / "label_free.json").write_text(json.dumps(rows))
    (tmp_path / "match_v.json").write_text(json.dumps({"bin_shares": [0.25, 0.25, 0.5, 0.0]}))
    assert c15.predicted_bins("label_free", "v") == {("g", 0): 0, ("g", 1): 2}
    weights = c15.window_weights("label_free", instrument="v")
    assert np.isclose(weights[("g", 0)], 0.5) and np.isclose(weights[("g", 1)], 1.0)
