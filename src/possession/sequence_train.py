"""Training of the temporal network of `sequence.py` (task 13). Needs torch, which the submission
does not: run it as

    uv run --with torch==2.15.0 possession train-sequence [--folds] [--final]

`--folds` trains, for each noise seed and each IDSSE match, a network on the other six (in the
ball regime chosen by gate G2, `sequence.REGIME`) and stores its predictions for the held-out match
in both validation input regimes: these are what the IDSSE suites score. `--final` trains on all
seven matches, both noise seeds, and writes models/sequence/<key>.npz with its sha256. Nothing is
tuned on any evaluation set: the architecture (`sequence.ARCH`) and the training settings
(`TRAIN`) were fixed in tasks/13-sequence-model.md before any network was trained.

Training is on the CPU with a fixed seed, a fixed number of threads and deterministic algorithms.
Inference is never torch: the exported weights run in `sequence.py` (numpy float64), and every
export is checked against torch's own float64 output on a stored input (`sequence.check`).
"""

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np

from possession import regime as regime_mod
from possession import sequence, supervised

TRAIN = {
    "crop": 600,  # 2 minutes at 5 Hz per training window
    "batch": 16,
    "epochs": 30,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "seed": 0,
    "threads": 8,
    "flag_share": 0.05,  # a missing flag for features NaN on more than 5% of training rows
    "scaler_stride": 7,
}
LABEL_CODES = {"H": 0, "A": 1, "D": 2, "X": 3}  # the package's label codes


def _torch():
    import torch

    torch.manual_seed(TRAIN["seed"])
    torch.set_num_threads(TRAIN["threads"])
    torch.use_deterministic_algorithms(True)
    return torch


def build(n_in: int):
    """The network of `sequence.ARCH` in torch: 1x1 projection, residual dilated blocks, 2 heads."""
    torch = _torch()
    nn = torch.nn
    arch = sequence.ARCH

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            width, taps = arch["width"], arch["kernel"]
            self.proj = nn.Conv1d(n_in, width, 1)
            self.blocks = nn.ModuleList(
                nn.Conv1d(width, width, taps, dilation=d, padding=d * (taps - 1) // 2)
                for d in arch["dilations"]
            )
            self.head = nn.Conv1d(width, 2, 1)

        def forward(self, x):  # (B, C, T) -> (B, 2, T)
            h = torch.relu(self.proj(x))
            for conv in self.blocks:
                h = h + torch.relu(conv(h))
            return self.head(h)

    return Net()


def sequences(prepared_lists) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(X at 5 Hz, labels, sample weights) per half and orientation of the prepared matches."""
    ds = sequence.ARCH["ds"]
    out = []
    for prepared in prepared_lists:
        for d in prepared:
            m = d["m"]
            label = np.asarray(m["label"])
            weight = supervised.training_weights(label)  # the same for both orientations
            views = [(d["X"], label)]
            if d.get("Xs") is not None:
                views.append((d["Xs"], np.asarray(d["ys"])))
            for x, y in views:
                for a, b in sequence._halves(np.asarray(m["half"])):
                    idx = np.arange(a, b, ds)
                    out.append((np.asarray(x[idx], np.float32), y[idx], weight[idx]))
    return out


def fit_scaler(seqs) -> dict:
    rows = np.concatenate([x for x, _, _ in seqs])[:: TRAIN["scaler_stride"]].astype(np.float64)
    missing = np.isnan(rows).mean(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nan_to_num(np.nanmean(rows, axis=0))
        std = np.nan_to_num(np.nanstd(rows, axis=0)) + 1e-6
    flag = np.flatnonzero(missing > TRAIN["flag_share"])
    return {"mean": mean, "std": std, "flag": flag, "clip": sequence.ARCH["clip"]}


def fit(seqs, scaler: dict, log=print):
    """Train one network on `seqs` (raw features; scaled here). Deterministic on the CPU."""
    torch = _torch()
    functional = torch.nn.functional
    rng = np.random.default_rng(TRAIN["seed"])
    z = [sequence.scale(x, scaler).astype(np.float32) for x, _, _ in seqs]
    n_in, crop, batch = z[0].shape[1], TRAIN["crop"], TRAIN["batch"]
    net = build(n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])
    per_epoch = sum(max(1, int(np.ceil(len(v) / crop))) for v in z)
    steps = TRAIN["epochs"] * int(np.ceil(per_epoch / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=TRAIN["lr"], total_steps=steps)
    started = time.time()
    for epoch in range(TRAIN["epochs"]):
        items = [
            (s, int(rng.integers(0, max(len(v) - crop, 0) + 1)))
            for s, v in enumerate(z)
            for _ in range(max(1, int(np.ceil(len(v) / crop))))
        ]
        order = rng.permutation(len(items))
        net.train()
        total = 0.0
        for i in range(0, len(order), batch):
            chosen = [items[j] for j in order[i : i + batch]]
            xb = np.zeros((len(chosen), n_in, crop), np.float32)
            yb = np.full((len(chosen), crop), LABEL_CODES["X"], np.int64)
            wb = np.zeros((len(chosen), crop), np.float32)
            for r, (s, start) in enumerate(chosen):
                part = z[s][start : start + crop]
                n = len(part)
                xb[r, :, :n] = part.T
                yb[r, :n] = seqs[s][1][start : start + n]
                wb[r, :n] = seqs[s][2][start : start + n]
            x, y, w = torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(wb)
            out = net(x)
            graded = y != LABEL_CODES["X"]
            live = (y == LABEL_CODES["H"]) | (y == LABEL_CODES["A"])
            loss = 0.0
            for head, mask, target in (
                (0, graded, y == LABEL_CODES["D"]),
                (1, live, y == LABEL_CODES["H"]),
            ):
                if mask.any():
                    term = functional.binary_cross_entropy_with_logits(
                        out[:, head][mask], target[mask].float(), weight=w[mask], reduction="sum"
                    )
                    loss = loss + term / w[mask].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(chosen)
        log(f"  epoch {epoch + 1} loss {total / len(items):.4f} ({time.time() - started:.0f} s)")
    return net.eval()


def export(net, scaler: dict, path: Path, meta: dict) -> dict:
    """Write the network as float64 numpy arrays with its scaler and a reference input/output
    from torch in float64; refuse to write if numpy does not reproduce torch (gate G4)."""
    torch = _torch()
    arrays = {
        "mean": np.asarray(scaler["mean"], np.float64),
        "std": np.asarray(scaler["std"], np.float64),
        "flag": np.asarray(scaler["flag"], np.int64),
        "proj_w": net.proj.weight.detach().double().numpy().transpose(2, 1, 0),
        "proj_b": net.proj.bias.detach().double().numpy(),
        "head_w": net.head.weight.detach().double().numpy().transpose(2, 1, 0),
        "head_b": net.head.bias.detach().double().numpy(),
    }
    for j, conv in enumerate(net.blocks):
        arrays[f"w{j}"] = conv.weight.detach().double().numpy().transpose(2, 1, 0)
        arrays[f"b{j}"] = conv.bias.detach().double().numpy()
    n_in = arrays["proj_w"].shape[1]
    ref = np.random.default_rng(12345).normal(size=(400, n_in))
    double = build(n_in).double()
    double.load_state_dict({k: v.double() for k, v in net.state_dict().items()})
    with torch.no_grad():
        ref_out = double(torch.from_numpy(ref.T[None].copy()))[0].numpy().T
    arrays["ref_input"], arrays["ref_output"] = ref, np.ascontiguousarray(ref_out)
    full_meta = {
        "ds": sequence.ARCH["ds"],
        "clip": sequence.ARCH["clip"],
        "dilations": list(sequence.ARCH["dilations"]),
        "train": TRAIN,
        "torch": torch.__version__,
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.npz")
    np.savez(partial, meta=np.array(json.dumps(full_meta, sort_keys=True)), **arrays)
    loaded = sequence.load(partial)
    error = sequence.check(loaded)
    if error > 1e-9:
        partial.unlink()
        raise RuntimeError(f"numpy inference differs from torch by {error:.3g} (gate G4)")
    partial.replace(path)
    return sequence.load(path)


def run_key(labels: str = "raw", regime: str = sequence.REGIME) -> str:
    """Changes with the architecture, the training settings, the ball regime and the code."""
    parts = [
        json.dumps(sequence.ARCH, sort_keys=True),
        json.dumps(TRAIN, sort_keys=True),
        labels,
        regime,
        regime_mod.regime_key(regime),
        inspect.getsource(sequence),
        inspect.getsource(inspect.getmodule(run_key)),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _prepared(labels: str, seed: int, regime: str) -> dict:
    parts = regime_mod.MIXED.get(regime, (regime,))
    return {part: supervised.idsse_prepared(labels, seed, part) for part in parts}


def folds(labels: str = "raw", seeds=(0, 1), regime: str = sequence.REGIME, log=print) -> None:
    """Out-of-fold predictions of every IDSSE match, both noise seeds, both input regimes."""
    key = run_key(labels, regime)
    for seed in seeds:
        prepared = _prepared(labels, seed, regime)
        parts = list(prepared)
        inputs = {}
        for name, part in supervised.INPUT_REGIMES.items():
            inputs[name] = prepared.get(part) or supervised.idsse_prepared(labels, seed, part)
        names = [str(d["m"]["name"]) for d in prepared[parts[0]]]
        for k, name in enumerate(names):
            targets = {inp: sequence.fold_path(key, inp, name) for inp in inputs}
            if all(path.is_file() for path in targets.values()):
                log(f"fold {name}: cached")
                continue
            train = [[d for j, d in enumerate(prepared[p]) if j != k] for p in parts]
            seqs = sequences(train)
            scaler = fit_scaler(seqs)
            log(f"fold {name} (seed {seed}): {len(seqs)} sequences, {scaler['mean'].size} features")
            net = export(
                fit(seqs, scaler, log),
                scaler,
                sequence.FOLDS_DIR / key / "nets" / f"{name}.npz",
                {"fold": name, "seed": seed, "regime": regime, "labels": labels},
            )
            for inp, path in targets.items():
                entry = inputs[inp][k]
                assert str(entry["m"]["name"]) == name
                out = sequence.logits(net, entry["X"], np.asarray(entry["m"]["half"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, out)


def final(labels: str = "raw", regime: str = sequence.REGIME, log=print) -> Path:
    """The network on all seven IDSSE matches and both noise seeds, written to models/sequence/."""
    key = run_key(labels, regime)
    seqs = []
    for seed in (0, 1):
        part = regime_mod.seed_regime(regime, seed)
        seqs.extend(sequences([supervised.idsse_prepared(labels, seed, part)]))
    scaler = fit_scaler(seqs)
    log(f"final network: {len(seqs)} sequences")
    path = sequence.MODELS_DIR / f"{key}.npz"
    export(fit(seqs, scaler, log), scaler, path, {"fold": None, "regime": regime, "labels": labels})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (sequence.MODELS_DIR / "sha256").write_text(f"{digest}  {path.name}\n")
    return path
