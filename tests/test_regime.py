import numpy as np
import pytest

from possession import regime


def _container(n=20000, seed=0):
    """A package-style container: play in both halves, a stoppage every 40 s lasting 8-20 s."""
    rng = np.random.default_rng(seed)
    label = np.where(rng.random(n) < 0.5, regime.H, regime.A).astype(np.int8)
    label = np.repeat(label[::250], 250)[:n]
    half = np.ones(n, dtype=np.int8)
    half[: n // 2] = 1
    half[n // 2 :] = 2
    half[:100] = 0
    label[:100] = 3
    for start in range(500, n - 600, 1000):
        label[start : start + int(rng.integers(200, 500))] = regime.D
    return {
        "bx": rng.uniform(0, 105, n).astype(np.float32),
        "by": rng.uniform(0, 68, n).astype(np.float32),
        "px": rng.uniform(0, 105, (n, 4)).astype(np.float32),
        "py": rng.uniform(0, 68, (n, 4)).astype(np.float32),
        "team": np.array([0, 0, 1, 1], dtype=np.int8),
        "half": half,
        "label": label,
        "pitch": np.array([105.0, 68.0]),
        "name": "SYNTH",
    }


@pytest.mark.parametrize("name", sorted(regime.REGIMES))
def test_automatic_ball_loses_the_ball_in_stoppages(name):
    m = _container()
    out = regime.automatic_ball(m, name, seed=1)
    for key in ("px", "py", "team", "half", "label"):
        assert np.array_equal(out[key], m[key]), key
    missing = np.isnan(out["bx"])
    assert np.array_equal(missing, np.isnan(out["by"]))
    play = m["half"] > 0
    dead = (m["label"] == regime.D) & play
    live = np.isin(m["label"], (regime.H, regime.A)) & play
    # kept in view only at the start of some stoppages (up to 6 s of these 8-20 s ones)
    assert missing[dead].mean() > 0.7
    assert 0.1 < missing[live].mean() < 0.45
    assert not np.isnan(m["bx"]).any()  # the input is not modified


def test_automatic_ball_is_deterministic_per_seed():
    m = _container()
    a = regime.automatic_ball(m, "auto_lf", seed=3)
    b = regime.automatic_ball(m, "auto_lf", seed=3)
    c = regime.automatic_ball(m, "auto_lf", seed=4)
    assert np.array_equal(np.isnan(a["bx"]), np.isnan(b["bx"]))
    assert not np.array_equal(np.isnan(a["bx"]), np.isnan(c["bx"]))


def test_regimes_per_seed_keys_and_params():
    assert (
        regime.seed_regime("mix_lf", 0) == "none" and regime.seed_regime("mix_lf", 1) == "auto_lf"
    )
    assert regime.seed_regime("auto_ml", 1) == "auto_ml"
    with pytest.raises(KeyError):
        regime.seed_regime("nope", 0)
    keys = {regime.regime_key(r) for r in ("auto_lf", "auto_ml", "mix_lf")}
    assert len(keys) == 3 and regime.regime_key("auto_lf") == regime.regime_key("auto_lf")
    params = {"ball_missing": 0.407, "player_visible": 0.93}
    assert regime.regime_params(params, "none", 0) is params
    assert regime.regime_params(params, "mix_lf", 0) is params
    tagged = regime.regime_params(params, "auto_ml", 0)
    assert tagged["ball_missing"] == 0.0 and tagged["regime"] == regime.regime_key("auto_ml")
    assert tagged["player_visible"] == 0.93
    m = _container(n=4000)
    assert regime.regime_inputs([m], "none", 0)[0] is m
    assert np.isnan(regime.regime_inputs([m], "auto_lf", 0)[0]["bx"]).any()
