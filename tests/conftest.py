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


def fake_probabilities(
    match: Match,
    labels: str = "raw",
    data: str = "idsse",
    regime: str = "none",
    tta: bool = False,
    features: str = "v1",
) -> np.ndarray:
    """(n, 3) P(H), P(A), P(D) standing in for the supervised model: a deterministic function of
    the tracking (dead where the ball is not tracked, else the team of the nearest player)."""
    from possession import features

    _, team = features.nearest_player(match)
    p_d = np.where(match.ball_seen, 0.1, 0.8)
    home = np.where(team == HOME, 0.8, 0.2)
    return np.stack([(1 - p_d) * home, (1 - p_d) * (1 - home), p_d], axis=1)


def fake_sequence_logits(match: Match) -> np.ndarray:
    """(n, 2) logits (z_dead, z_home) standing in for the temporal network: the log-odds of
    `fake_probabilities`, so both learned sources tell the same story in tests."""
    p = fake_probabilities(match)
    alive = 1.0 - p[:, 2]
    home = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)
    return np.stack([np.log(p[:, 2] / alive), np.log(home / (1.0 - home))], axis=1)


@pytest.fixture(autouse=True)
def stub_supervised_model(request, monkeypatch):
    """The supervised model needs the match data and 2.7 GB of open data, so unit tests replace
    its probabilities with `fake_probabilities`. The methods built on it (`ball_state.supervised`,
    `team.supervised`, the hybrid config) are then covered on a fresh clone like every other
    method; what is not covered here is the model itself. Mark a test `needs_model` to keep the
    real one."""
    if request.node.get_closest_marker("needs_model"):
        return
    from possession import sequence, supervised

    monkeypatch.setattr(supervised, "probabilities", fake_probabilities)
    monkeypatch.setattr(sequence, "predictions", fake_sequence_logits)
    from possession import sequence_seeds

    monkeypatch.setattr(sequence_seeds, "network", fake_seed_network)
    monkeypatch.setattr(sequence_seeds, "_features", fake_seed_features)


def tiny_network(n_in: int, seed: int = 1) -> dict:
    """A small random network in the layout of `sequence.load` (2 blocks, width 4)."""
    rng = np.random.default_rng(seed)
    width = 4
    net = {
        "ds": 5,
        "clip": 5.0,
        "dilations": (1, 2),
        "mean": np.zeros(n_in),
        "std": np.ones(n_in),
        "flag": np.array([0]),
        "proj_w": rng.normal(size=(1, n_in + 1, width)),
        "proj_b": rng.normal(size=width),
        "head_w": rng.normal(size=(1, width, 2)),
        "head_b": rng.normal(size=2),
    }
    for j in range(2):
        net[f"w{j}"] = rng.normal(size=(3, width, width)) * 0.3
        net[f"b{j}"] = rng.normal(size=width) * 0.1
    return net


def fake_seed_network(seed: int) -> dict:
    """Stands in for `sequence_seeds.network`: a tiny network per training seed."""
    return tiny_network(3, seed=100 + int(seed))


def fake_seed_features(match: Match, swapped: bool) -> np.ndarray:
    """Stands in for the package's features: ball position (mirrored in the swapped view) and
    whether the ball is tracked."""
    x, y = match.ball[:, 0] / 50.0, match.ball[:, 1] / 30.0
    if swapped:
        x = -x
    return np.column_stack([x, y, match.ball_seen.astype(float)])
