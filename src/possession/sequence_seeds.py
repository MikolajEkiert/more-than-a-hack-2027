"""Several initialisations of the temporal network of `sequence.py`, averaged (task 19).

The committed network (models/sequence/, training seed 0) is one draw of its initialisation and of
the order of its training windows. This module trains the same network, on the same data with the
same settings, under another training seed, and averages the networks' stoppage log-odds; it can
also average each network's prediction with that of the same match with home and away exchanged
(a test-time view). Leading teams of the challenge average several seeds and views of their
temporal networks (Team Sigue: three seeds and four views; TercerHombre: a multi-seed MS-TCN);
the idea is theirs, this is our own implementation on our own network.

`sequence.py` and `sequence_train.py` are not edited (their source is part of every key): the
training seed is swapped in by `train_seed`, which replaces `sequence_train.TRAIN` with a copy for
the duration of a training. The key of a network (`run_key`) therefore changes with the seed, and
the seed-0 key is the committed network's. Extra final networks live in models/sequence_seeds/
with their own sha256 listing, so the committed network and its listing are never touched.

Inference is numpy float64 (`sequence.logits`), never torch. A training match (IDSSE) is only ever
scored by networks trained without it: its out-of-fold predictions are written by
`sequence_train.folds` under each seed's key, the swapped views by `swap_folds`.
"""

import hashlib
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np

from possession import features, sequence
from possession._paths import REPO_ROOT
from possession.match import Match

SEEDS_DIR = REPO_ROOT / "models" / "sequence_seeds"  # <key>.npz of seeds other than 0, `sha256`
SWAP = "_swap"  # suffix of the input-regime folder holding out-of-fold swapped-view predictions


# -- keys and training -------------------------------------------------------------------------


@contextmanager
def train_seed(seed: int):
    """`sequence_train` with TRAIN["seed"] = `seed` inside the block (a copy; restored after)."""
    from possession import sequence_train

    original = sequence_train.TRAIN
    sequence_train.TRAIN = {**original, "seed": int(seed)}
    try:
        yield sequence_train
    finally:
        sequence_train.TRAIN = original


def run_key(seed: int) -> str:
    """The key of the network trained with training seed `seed` (seed 0: the committed one)."""
    with train_seed(seed) as st:
        return st.run_key()


def train_folds(seed: int, log=print) -> None:
    """Out-of-fold IDSSE predictions of the network with training seed `seed` (needs torch)."""
    _storage()
    with train_seed(seed) as st:
        st.folds(log=log)


def train_final(seed: int, log=print) -> Path:
    """The network with training seed `seed` on all seven IDSSE matches and both noise seeds,
    written to models/sequence_seeds/<key>.npz and listed in its sha256 file (needs torch).

    The same steps as `sequence_train.final`, which would overwrite the committed network's
    listing; seed 0 is refused (that network is the committed one).
    """
    if int(seed) == 0:
        raise ValueError("seed 0 is the committed network: `possession train-sequence --final`")
    from possession import regime as regime_mod
    from possession import supervised

    _storage()
    with train_seed(seed) as st:
        key = st.run_key()
        seqs = []
        for noise in (0, 1):
            part = regime_mod.seed_regime(sequence.REGIME, noise)
            seqs.extend(st.sequences([supervised.idsse_prepared("raw", noise, part)]))
        scaler = st.fit_scaler(seqs)
        log(f"final network, training seed {seed}: {len(seqs)} sequences")
        path = SEEDS_DIR / f"{key}.npz"
        meta = {"fold": None, "regime": sequence.REGIME, "labels": "raw"}
        st.export(st.fit(seqs, scaler, log), scaler, path, meta)
    listing = SEEDS_DIR / "sha256"
    entries = _listing(listing)
    entries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    listing.write_text("".join(f"{entries[k]}  {k}\n" for k in sorted(entries)))
    return path


def _storage() -> None:
    """Point the package's storage at .cache/ before anything imports it (`supervised._package`)."""
    from possession import supervised

    supervised._package()


def _listing(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = (line.split() for line in path.read_text().splitlines() if line)
    return {name: digest for digest, name in rows}


@cache
def network(seed: int) -> dict:
    """The final network of training seed `seed`, its sha256 checked against its listing."""
    if int(seed) == 0:
        return sequence.final()
    key = run_key(seed)
    path = SEEDS_DIR / f"{key}.npz"
    expected = _listing(SEEDS_DIR / "sha256").get(path.name)
    if expected is None or not path.is_file():
        raise FileNotFoundError(
            f"no network for training seed {seed} ({path.name}); run "
            f"`uv run --with torch==2.15.0 possession train-sequence --seed {seed} --final`"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path}: sha256 does not match {SEEDS_DIR / 'sha256'}")
    return sequence.load(path)


# -- the swapped view --------------------------------------------------------------------------


def swapped_features_of(match: Match) -> np.ndarray:
    """(T, K) float16: the package's features of `match` with home and away exchanged, from the
    same disk cache as the LightGBM's test-time view (`supervised._predict` with tta)."""
    import os

    from possession import supervised

    _, feature_fn, _ = supervised._setup()
    supervised._package()
    from solution import cache as package_cache
    from solution import container as package_container
    from solution import features as package_features

    m = supervised.container(match)
    view = package_container.swap_teams(m)
    if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
        return feature_fn(view)[0].astype(package_cache.FEATURE_DTYPE)
    key = package_cache.digest("match", package_cache.fingerprint(m))
    return package_features.cached(feature_fn, view, "swap", key)[0]


def swap_folds(seed: int, log=print) -> None:
    """Out-of-fold predictions of the swapped view of every IDSSE match copy, from the fold
    networks of training seed `seed` (numpy only; the fold networks must exist)."""
    from possession import supervised

    _storage()
    key = run_key(seed)
    for noise in (0, 1):
        for inp, part in supervised.INPUT_REGIMES.items():
            todo = None
            for entry in _prepared_swap(noise, part):
                name = str(entry["m"]["name"])
                target = sequence.fold_path(key, inp + SWAP, name)
                if target.is_file():
                    continue
                net = sequence.load(sequence.FOLDS_DIR / key / "nets" / f"{name}.npz")
                half = np.asarray(entry["m"]["half"])
                # the stored prediction of the original view is reproduced: same copy, same net
                stored = np.load(sequence.fold_path(key, inp, name))
                if not np.array_equal(sequence.logits(net, entry["X"], half), stored):
                    raise RuntimeError(f"{name} ({inp}): the fold prediction is not reproduced")
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, sequence.logits(net, entry["Xs"], half))
                todo = (todo or 0) + 1
            log(f"seed {seed}, noise {noise}, {inp}: {todo or 0} swapped folds written")


def _prepared_swap(noise: int, part: str) -> list:
    from possession import supervised

    return supervised.idsse_prepared("raw", noise, part, swap=True)


# -- evidence ----------------------------------------------------------------------------------


def _closure(fn) -> dict:
    cells = fn.__closure__ or ()
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in cells), strict=True))


def _view_logits(match: Match, seed: int, swapped: bool) -> np.ndarray:
    """(n, 2) logits of network `seed` for `match` (its swapped view if `swapped`), out of fold
    for a training match, memoised on the match."""

    def compute():
        held = match._cache.get(sequence.HELD_OUT)
        qualname = getattr(held, "__qualname__", "")
        if held is not None and qualname.startswith("held_out."):
            free = _closure(held)  # an IDSSE copy: name and input regime of its fold files
            folder = free["input_regime"] + (SWAP if swapped else "")
            path = sequence.fold_path(run_key(seed), folder, free["name"])
            if not path.is_file():
                raise FileNotFoundError(
                    f"no out-of-fold prediction {path}; train the folds of seed {seed}"
                    + (" and run `sequence_seeds.swap_folds`" if swapped else "")
                )
            return np.load(path)
        if held is not None and not qualname.startswith("never_trained_on."):
            raise RuntimeError(f"unknown held-out attachment {qualname!r} on {match.game}")
        if held is None and match.game.startswith(sequence.TRAINING_PREFIXES):
            raise RuntimeError(
                f"{match.game} is named like a training match of the sequence network and has no "
                "held-out prediction attached; score it through the validation conditions"
            )
        return sequence.logits(network(seed), _features(match, swapped), match.half)

    return features._memo(match, ("sequence_seed", int(seed), bool(swapped)), compute)


def _features(match: Match, swapped: bool) -> np.ndarray:
    """The network's input: the package's features of `match` or of its swapped view."""
    from possession import supervised

    return swapped_features_of(match) if swapped else supervised.features_of(match)


def z_dead(match: Match, seeds=(0, 1, 2), tta: bool = False) -> np.ndarray:
    """(n,) float64: the mean stoppage log-odds over the networks of `seeds` (and their swapped
    views with `tta`)."""
    views = [False, True] if tta else [False]
    parts = [_view_logits(match, int(s), v)[:, 0] for s in seeds for v in views]
    return np.mean(parts, axis=0)
