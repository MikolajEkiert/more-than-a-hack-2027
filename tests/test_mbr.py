"""The minimum-Bayes-risk decoder (task 14): contract, determinism, and its limit."""

import numpy as np

from possession.stages import decode, mbr
from tests.conftest import make_match


def _evidence(match, seed=0):
    rng = np.random.default_rng(seed)
    n = match.n_frames
    block = np.repeat(rng.random(n // 150 + 1), 150)[:n]
    p_alive = np.clip(0.15 + 0.8 * (block > 0.25) + rng.normal(0, 0.05, n), 0.01, 0.99)
    p_home = np.clip(np.where(block > 0.6, 0.85, 0.2) + rng.normal(0, 0.1, n), 0.01, 0.99)
    p_home[rng.random(n) < 0.2] = np.nan
    return p_alive, p_home


def test_labels_and_determinism():
    match = make_match(n=4000)
    p_alive, p_home = _evidence(match)
    a = mbr.mbr(match, p_alive, p_home, n_samples=16)
    b = mbr.mbr(match, p_alive, p_home, n_samples=16)
    assert a.shape == (match.n_frames,)
    assert set(np.unique(a[match.graded])) <= {"H", "A", "D"}
    assert (a == b).all()


def test_sharp_posterior_gives_the_most_likely_path():
    match = make_match(n=4000)
    p_alive, p_home = _evidence(match, seed=1)
    viterbi = decode.switch_cost(
        match, p_alive, p_home, turnover_cost=5, stop_cost=12, restart_cost=12
    )
    sharp = mbr.mbr(match, p_alive, p_home, sharpness=50.0, n_samples=8, spell_cost=0.05)
    graded = match.graded
    assert np.mean(sharp[graded] == viterbi[graded]) > 0.99


def test_choose_takes_the_best_non_overlapping_set():
    cands = np.array([[0, 10, 0], [5, 20, 1], [10, 30, 0], [30, 40, 1]])
    value = np.array([0.5, 0.9, 0.5, 0.2])
    chosen = mbr.choose(cands, value)
    assert chosen.tolist() == [0, 2, 3]  # 0.5 + 0.5 + 0.2 = 1.2 > 0.9 + 0.2
