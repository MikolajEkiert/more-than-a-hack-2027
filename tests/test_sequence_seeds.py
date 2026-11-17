import dataclasses

import numpy as np
import pytest

from possession import sequence, sequence_seeds
from possession._paths import REPO_ROOT
from possession.stages import ball_state
from tests.conftest import fake_seed_features, fake_seed_network, make_match


def test_seed_zero_is_the_committed_network_and_train_is_restored():
    from possession import sequence_train

    before = sequence_train.TRAIN
    listing = (REPO_ROOT / "models" / "sequence" / "sha256").read_text().split()
    assert f"{sequence_seeds.run_key(0)}.npz" == listing[1]
    assert len({sequence_seeds.run_key(s) for s in (0, 1, 2)}) == 3
    with sequence_seeds.train_seed(5) as st:
        assert st.TRAIN["seed"] == 5 and st.TRAIN is not before
        assert {**st.TRAIN, "seed": 0} == before
    assert sequence_train.TRAIN is before and before["seed"] == 0


def test_final_of_seed_zero_is_refused():
    with pytest.raises(ValueError):
        sequence_seeds.train_final(0)


def _expected(match, seeds, tta):
    parts = []
    for s in seeds:
        for swapped in [False, True] if tta else [False]:
            x = fake_seed_features(match, swapped)
            parts.append(sequence.logits(fake_seed_network(s), x, match.half)[:, 0])
    return 1.0 / (1.0 + np.exp(np.mean(parts, axis=0)))


@pytest.mark.parametrize(
    "seeds,tta", [([0], False), ([0], True), ([0, 1, 2], False), ([0, 2], True)]
)
def test_ensemble_is_the_mean_log_odds_of_its_networks_and_views(seeds, tta):
    match = make_match(n=1200)
    got = ball_state.sequence_ensemble(match, seeds=seeds, tta=tta)
    np.testing.assert_allclose(got, _expected(match, seeds, tta), rtol=0, atol=1e-12)


def test_views_and_seeds_change_the_prediction():
    match = make_match(n=1200)
    one = ball_state.sequence_ensemble(match, seeds=[0])
    assert not np.allclose(one, ball_state.sequence_ensemble(match, seeds=[0, 1]))
    assert not np.allclose(one, ball_state.sequence_ensemble(match, seeds=[0], tta=True))


def _training_match(n=600):
    return dataclasses.replace(make_match(n=n), game="IDSSE-TEST", _cache={})


def test_a_training_match_is_read_from_its_out_of_fold_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sequence, "FOLDS_DIR", tmp_path)
    monkeypatch.setattr(sequence_seeds, "run_key", lambda seed: f"key{seed}")
    match = _training_match()
    match._cache[sequence.HELD_OUT] = sequence.held_out("IDSSE-TEST_deg0", "auto")
    rng = np.random.default_rng(0)
    stored = []
    for seed in (0, 1):
        for folder in ("auto", "auto" + sequence_seeds.SWAP):
            value = rng.normal(size=(600, 2))
            path = sequence.fold_path(f"key{seed}", folder, "IDSSE-TEST_deg0")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value)
            stored.append(value[:, 0])
    z = sequence_seeds.z_dead(match, seeds=(0, 1), tta=True)
    np.testing.assert_allclose(z, np.mean(stored, axis=0), atol=1e-12)
    with pytest.raises(FileNotFoundError):
        sequence_seeds.z_dead(match, seeds=(2,))


def test_a_training_match_without_held_out_prediction_is_an_error():
    with pytest.raises(RuntimeError):
        sequence_seeds.z_dead(_training_match(), seeds=(0,))
