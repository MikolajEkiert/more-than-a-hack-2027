"""Fusion of evidence sources: the arithmetic, the file reader and the blend methods."""

import numpy as np
import pytest

from possession import config as config_mod
from possession import fusion, pipeline
from possession.stages.ball_state import blend as blend_ball_state
from possession.stages.ball_state import stoppage_evidence
from possession.stages.team import blend as blend_team
from possession.stages.team import control_evidence, touch_episodes
from tests.conftest import make_match


def test_combine_adds_log_odds_and_skips_missing_votes():
    a = np.array([0.9, 0.9, np.nan, np.nan, 0.5])
    b = np.array([0.9, 0.1, 0.8, np.nan, 0.2])
    out = fusion.combine([a, b], [1.0, 1.0])
    assert out[0] == pytest.approx(0.81 / 0.82)  # two agreeing sources are surer than either
    assert out[1] == pytest.approx(0.5)  # opposite votes of equal strength cancel
    assert out[2] == pytest.approx(0.8)  # one source without a vote changes nothing
    assert np.isnan(out[3])  # nobody votes
    assert out[4] == pytest.approx(0.2)  # even odds add nothing

    half = fusion.combine([a, b], [1.0, 0.0])
    np.testing.assert_allclose(half[:2], [0.9, 0.9])  # weight 0 silences a source
    assert half[2] == pytest.approx(0.5) and np.isnan(half[3])
    assert fusion.combine([a], [2.0])[0] == pytest.approx(0.81 / 0.82)  # weight 2 = counted twice


def test_blend_methods_default_to_one_source_and_accept_tables():
    match = make_match()
    np.testing.assert_allclose(blend_ball_state(match), stoppage_evidence(match), atol=1e-5)
    np.testing.assert_allclose(blend_team(match), touch_episodes(match), atol=1e-5)

    sources = ["touch_episodes", {"method": "control_evidence", "control_radius_m": 3.0}]
    both = blend_team(match, sources=sources, weights=[1.0, 0.5])
    a, b = touch_episodes(match), control_evidence(match, control_radius_m=3.0)
    expected = fusion.combine([a.astype(float), b.astype(float)], [1.0, 0.5])
    np.testing.assert_allclose(both, expected, atol=1e-5)
    assert np.isnan(both).sum() <= min(np.isnan(a).sum(), np.isnan(b).sum())

    with pytest.raises(ValueError):
        blend_team(match, sources=["touch_episodes"], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        blend_team(match, sources=[{"method": "touch_episodes", "no_such_parameter": 1}])
    with pytest.raises(KeyError):
        blend_team(match, sources=["no_such_method"])


def test_evidence_from_files(tmp_path):
    match = make_match(n=400)
    n = match.n_frames
    p_d = np.linspace(0.0, 1.0, n)
    p_h = (1 - p_d) * 0.75
    p_a = (1 - p_d) * 0.25
    three = tmp_path / "SYNTH.npz"
    np.savez(three, p_h=p_h, p_a=p_a, p_d=p_d)
    template = str(tmp_path / "{game}.npz")  # {game} is the match's name
    alive = fusion.from_file(match, "ball_state", template)
    home = fusion.from_file(match, "team", template)
    np.testing.assert_allclose(alive, 1 - p_d)
    np.testing.assert_allclose(home[:-1], 0.75)
    assert np.isnan(home[-1])  # certainly dead: no opinion about the team

    own = tmp_path / "own.npz"
    np.savez(own, p_home=np.full(n, 0.3), mine=np.full(n, 0.6))
    np.testing.assert_allclose(fusion.from_file(match, "team", str(own)), 0.3)
    np.testing.assert_allclose(fusion.from_file(match, "team", str(own), key="mine"), 0.6)
    plain = tmp_path / "plain.npy"
    np.save(plain, np.full(n, 0.8))
    np.testing.assert_allclose(fusion.from_file(match, "ball_state", str(plain)), 0.8)

    fused = blend_team(match, sources=["touch_episodes", {"file": str(own)}], weights=[0.0, 1.0])
    np.testing.assert_allclose(fused, 0.3, atol=1e-5)  # a file is a source like any other

    np.save(tmp_path / "short.npy", np.full(n - 1, 0.8))
    with pytest.raises(ValueError):
        fusion.from_file(match, "ball_state", str(tmp_path / "short.npy"))
    np.save(tmp_path / "bad.npy", np.full(n, 1.5))
    with pytest.raises(ValueError):
        fusion.from_file(match, "ball_state", str(tmp_path / "bad.npy"))
    with pytest.raises(FileNotFoundError):
        fusion.from_file(match, "ball_state", str(tmp_path / "missing.npy"))


def test_a_blend_config_runs_through_the_pipeline():
    cfg = config_mod.parse(
        {
            "name": "blend_test",
            "ball_state": {"method": "blend", "sources": ["stoppage_evidence", "ball_visible"]},
            "team": {
                "method": "blend",
                "sources": ["touch_episodes", {"method": "control_evidence"}],
                "weights": [1.0, 0.5],
            },
            "decode": {"method": "switch_cost"},
        }
    )
    match = make_match()
    labels = pipeline.run(cfg, match).labels
    assert set(labels[match.graded]) <= {"H", "A", "D"} and set(labels[~match.graded]) <= {"X"}
    assert cfg.as_dict()["team"]["weights"] == [1.0, 0.5]


def test_degraded_matches_carry_their_condition_in_the_name():
    from possession import validation
    from tests.test_validation import _clean_match

    match, labels = _clean_match()
    degraded = validation.degrade(match, labels, seed=2)
    assert degraded.game == "SYNTH@label_free#2" and match.game == "SYNTH"
