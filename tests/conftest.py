"""Synthetic fixtures. Tests never need the match data, so they run in CI and on a fresh clone.

Tests that do need the real data are marked `needs_data` and skip themselves when it is absent.
"""

import numpy as np
import pytest

from possession.match import AWAY, HOME, Match


def make_match(n=3000, n_slots=8, seed=0, ungraded_head=200, ball_missing=0.3) -> Match:
    """A small, deterministic, plausible-looking match: players drift, the ball follows one of
    them, and the tracking drops the ball in contiguous gaps like the real ATD does."""
    rng = np.random.default_rng(seed)
    length, width = 105.3, 68.0
    team = np.array([HOME, AWAY] * (n_slots // 2), dtype=np.int8)

    steps = rng.normal(0, 0.3, size=(n, n_slots, 2))
    xy = np.cumsum(steps, axis=0) + rng.uniform([10, 10], [95, 58], size=(1, n_slots, 2))
    xy[..., 0] = np.clip(xy[..., 0], 0, length)
    xy[..., 1] = np.clip(xy[..., 1], 0, width)
    speed = np.abs(rng.normal(3, 1, size=(n, n_slots, 1)))
    players = np.concatenate([xy, speed], axis=2)

    carrier = np.repeat(rng.integers(0, n_slots, size=n // 100 + 1), 100)[:n]
    ball = np.concatenate([xy[np.arange(n), carrier] + 0.5, np.full((n, 1), 5.0)], axis=1)
    gaps = rng.random(n // 50 + 1) < ball_missing
    ball[np.repeat(gaps, 50)[:n]] = np.nan
    players[rng.random((n, n_slots)) < 0.05] = np.nan

    half = np.ones(n, dtype=np.int8)
    half[:ungraded_head] = 0
    return Match(
        game="SYNTH",
        fps=25,
        pitch=(length, width),
        ball=ball,
        players=players,
        team_of_track=team,
        half=half,
        tagging=None,
    )


@pytest.fixture
def match():
    return make_match()


def fake_probabilities(match: Match, labels: str = "raw") -> np.ndarray:
    """(n, 3) P(H), P(A), P(D) standing in for the supervised model: a deterministic function of
    the tracking (dead where the ball is not tracked, else the team of the nearest player)."""
    from possession import features

    _, team = features.nearest_player(match)
    p_d = np.where(match.ball_seen, 0.1, 0.8)
    home = np.where(team == HOME, 0.8, 0.2)
    return np.stack([(1 - p_d) * home, (1 - p_d) * (1 - home), p_d], axis=1)


@pytest.fixture(autouse=True)
def stub_supervised_model(request, monkeypatch):
    """The supervised model needs the match data and 2.7 GB of open data, so unit tests replace
    its probabilities with `fake_probabilities`. The methods built on it (`ball_state.supervised`,
    `team.supervised`, the hybrid config) are then covered on a fresh clone like every other
    method; what is not covered here is the model itself. Mark a test `needs_model` to keep the
    real one."""
    if request.node.get_closest_marker("needs_model"):
        return
    from possession import supervised

    monkeypatch.setattr(supervised, "probabilities", fake_probabilities)
