"""Every registered method honours its stage contract. New methods are covered automatically:
register it, and these tests run it with its default parameters on the synthetic match."""

import numpy as np
import pytest

from possession import registry
from tests.conftest import make_match


def _ids(method):
    return f"{method.stage}.{method.name}"


def _inputs(match):
    p_alive = registry.methods("ball_state")[0].fn(match)
    p_home = registry.methods("team")[0].fn(match)
    return p_alive, p_home


@pytest.mark.parametrize("method", registry.methods("ball_state"), ids=_ids)
def test_ball_state_contract(method, match):
    out = np.asarray(method.fn(match))
    assert out.shape == (match.n_frames,)
    assert not np.isnan(out).any(), "ball_state may not return NaN"
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("method", registry.methods("team"), ids=_ids)
def test_team_contract(method, match):
    out = np.asarray(method.fn(match), dtype=float)
    assert out.shape == (match.n_frames,)
    finite = out[~np.isnan(out)]
    assert finite.size, "a team method that never has evidence is not doing anything"
    assert finite.min() >= 0 and finite.max() <= 1


@pytest.mark.parametrize("method", registry.methods("decode"), ids=_ids)
def test_decode_contract(method, match):
    p_alive, p_home = _inputs(match)
    out = np.asarray(method.fn(match, p_alive, p_home))
    assert out.shape == (match.n_frames,)
    assert set(np.unique(out)) <= {"H", "A", "D", "X"}


@pytest.mark.parametrize("method", registry.methods(), ids=_ids)
def test_deterministic(method):
    """Same input, same output, twice -- RULES.md 3 needs the exact same file."""
    a, b = make_match(seed=1), make_match(seed=1)
    if method.stage == "decode":
        out_a, out_b = method.fn(a, *_inputs(a)), method.fn(b, *_inputs(b))
    else:
        out_a, out_b = method.fn(a), method.fn(b)
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


@pytest.mark.parametrize("method", registry.methods(), ids=_ids)
def test_does_not_mutate_match(method, match):
    before = (match.ball.copy(), match.players.copy(), match.half.copy())
    if method.stage == "decode":
        method.fn(match, *_inputs(match))
    else:
        method.fn(match)
    for original, now in zip(before, (match.ball, match.players, match.half), strict=True):
        np.testing.assert_array_equal(original, now)
