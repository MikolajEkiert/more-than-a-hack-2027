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
