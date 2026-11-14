"""A temporal network over the package's per-frame features, as stage evidence (task 13).

Two heads over the same dilated temporal convolution network (TCN): `z_dead`, the log-odds that
play is stopped, and `z_home`, the log-odds that the home team has the ball given play. The
network sees about 51 s of context on each side (5 Hz, 7 residual blocks, dilations 1 to 64),
which is what the frame-by-frame LightGBM lacks: stoppages are recognised from what happens
around them (docs/FUSION.md section 10, task 11).

This module is inference only, in numpy float64, with every convolution written as an explicit
sum over input channels and taps: no BLAS, a fixed summation order, so the result does not depend
on the machine. The training code (`sequence_train.py`) needs torch; this module does not.

A match the network could have been trained on (named like an IDSSE or PFF training match) is
only ever scored through a function attached under `HELD_OUT` (`supervised.idsse_conditions`,
`supervised.pff_conditions`), exactly as for the LightGBM (`supervised.probabilities`).
"""

import json
from functools import cache
from pathlib import Path

import numpy as np

from possession import features
from possession._paths import CACHE_DIR, REPO_ROOT
from possession.match import Match

ARCH = {"ds": 5, "width": 64, "kernel": 3, "dilations": (1, 2, 4, 8, 16, 32, 64), "clip": 5.0}
REGIME = "mix_lf"  # the training regime chosen by gate G2 (docs/FUSION.md section 10)
MODELS_DIR = REPO_ROOT / "models" / "sequence"  # <key>.npz, and `sha256` listing each file's hash
FOLDS_DIR = CACHE_DIR / "sequence"  # out-of-fold predictions, written by `sequence_train.folds`
HELD_OUT = "sequence_held_out"  # Match._cache key: () -> (n, 2) logits from a network without it
TRAINING_PREFIXES = ("IDSSE-", "PFF-")


# -- the network -----------------------------------------------------------------------------


def scale(x: np.ndarray, net: dict) -> np.ndarray:
    """Standardised features clipped to +-clip, NaN as 0, plus a missing flag per listed column."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - net["mean"]) / net["std"]
    z = np.clip(z, -net["clip"], net["clip"])
    flags = np.isnan(x[:, net["flag"]]).astype(np.float64)
    return np.concatenate([np.nan_to_num(z, nan=0.0), flags], axis=1)


def conv_same(x: np.ndarray, w: np.ndarray, b: np.ndarray, dilation: int) -> np.ndarray:
    """1-D convolution with zero 'same' padding: x (T, Cin), w (K, Cin, Cout), b (Cout,).

    Written as an explicit sum over taps and input channels, in that order, so that every output
    is the same float64 sum on every machine.
    """
    steps, cin = x.shape
    taps = w.shape[0]
    pad = dilation * (taps - 1) // 2
    padded = np.zeros((steps + 2 * pad, cin))
    padded[pad : pad + steps] = x
    out = np.empty((steps, w.shape[2]))
    out[:] = b
    for k in range(taps):
        shifted = padded[k * dilation : k * dilation + steps]
        for i in range(cin):
            out += shifted[:, i : i + 1] * w[k, i]
    return out


def forward(net: dict, z: np.ndarray) -> np.ndarray:
    """(T5, 2) logits (z_dead, z_home) for one half at 5 Hz."""
    h = np.maximum(conv_same(z, net["proj_w"], net["proj_b"], 1), 0.0)
    for j, dilation in enumerate(net["dilations"]):
        h = h + np.maximum(conv_same(h, net[f"w{j}"], net[f"b{j}"], int(dilation)), 0.0)
    return conv_same(h, net["head_w"], net["head_b"], 1)


def _halves(half: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for number in (1, 2):
        starts, stops = features.runs(half == number)
        out.extend(zip(starts.tolist(), stops.tolist(), strict=True))
    return out


def logits(net: dict, x: np.ndarray, half: np.ndarray) -> np.ndarray:
    """(n, 2) logits at 25 fps from the full-length feature matrix `x`; 0 outside the halves.

    Each half is run at 5 Hz (every `ds`-th frame) and interpolated back linearly.
    """
    ds = int(net["ds"])
    out = np.zeros((len(half), 2))
    for a, b in _halves(half):
        idx = np.arange(a, b, ds)
        low = forward(net, scale(x[idx], net))
        frames = np.arange(a, b)
        for c in (0, 1):
            out[a:b, c] = np.interp(frames, idx, low[:, c])
    return out


# -- weights ---------------------------------------------------------------------------------


def load(path: Path) -> dict:
    """A network exported by `sequence_train.export`: float64 arrays and its metadata."""
    with np.load(path, allow_pickle=False) as data:
        net = {k: np.asarray(data[k]) for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    net.update(meta)
    net["dilations"] = tuple(int(d) for d in meta["dilations"])
    net["flag"] = np.asarray(net["flag"], dtype=np.int64)
    return net


def check(net: dict) -> float:
    """Max |numpy - torch| on the reference input stored at export (torch's float64 output)."""
    got = forward(net, net["ref_input"])
    return float(np.max(np.abs(got - net["ref_output"])))


@cache
def final(key: str | None = None) -> dict:
    """The committed network (models/sequence/<key>.npz), its sha256 checked against the listing."""
    import hashlib

    listing = MODELS_DIR / "sha256"
    if not listing.is_file():
        raise FileNotFoundError(f"no committed sequence network: {listing} is missing")
    expected = dict(line.split()[::-1] for line in listing.read_text().splitlines() if line)
    if key is None:
        if len(expected) != 1:
            raise ValueError(f"{listing} lists {len(expected)} networks; name one")
        key = next(iter(expected)).removesuffix(".npz")
    path = MODELS_DIR / f"{key}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(path.name):
        raise ValueError(f"{path}: sha256 does not match {listing}")
    return load(path)


# -- evidence for the stages ----------------------------------------------------------------


def predictions(match: Match) -> np.ndarray:
    """(n, 2) logits (z_dead, z_home) for `match`, memoised; out of fold for a training match."""

    def compute():
        held_out = match._cache.get(HELD_OUT)
        if held_out is not None:
            return np.asarray(held_out(), dtype=float)
        if match.game.startswith(TRAINING_PREFIXES):
            raise RuntimeError(
                f"{match.game} is named like a training match of the sequence network and has no "
                "held-out prediction attached; score it through the validation conditions"
            )
        from possession import supervised

        return logits(final(), supervised.features_of(match), match.half)

    return features._memo(match, "sequence", compute)


def p_alive(match: Match) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(predictions(match)[:, 0]))


def p_home(match: Match) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-predictions(match)[:, 1]))


# -- out of fold ------------------------------------------------------------------------------


def fold_path(key: str, input_regime: str, name: str) -> Path:
    return FOLDS_DIR / key / input_regime / f"{name}.npy"


def held_out(name: str, input_regime: str):
    """() -> the out-of-fold logits of the IDSSE match copy `name` (written by
    `sequence_train.folds`, never computed here, so validation never needs torch)."""

    def get():
        from possession import sequence_train

        path = fold_path(sequence_train.run_key(), input_regime, name)
        if not path.is_file():
            raise FileNotFoundError(
                f"no out-of-fold prediction of the sequence network for {name}; run "
                "`uv run --with torch==2.15.0 possession train-sequence --folds`"
            )
        return np.load(path)

    return get


def never_trained_on(match: Match):
    """() -> logits of the committed network for a match it was never trained on (PFF test)."""

    def get():
        from possession import supervised

        return logits(final(), supervised.features_of(match), match.half)

    return get
