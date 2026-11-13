"""Temporal network on the per-frame features: bi-GRU at 5 Hz, ensembled with LightGBM.

Every top team in research/COMPETITORS.md adds a temporal network (TCN / MS-TCN / 1D-CNN / bi-GRU)
to gradient boosting. Here: features (the same 203, standardised, NaN -> 0 + missing indicators)
-> linear -> 2-layer bi-GRU -> 3 classes, trained on 60 s windows with boundary-weighted
cross-entropy, inferred over whole halves, interpolated back to 25 fps.

Reproducibility: GPU training is not bit-exact, so the FINAL network's weights are saved to
models/ and the submission is produced from that file on CPU (deterministic). The training code
that produced it is this module.
"""

import os

import numpy as np

from solution import cache, container as C, evaluate_external as ev, features

# Params
DS = 5                     # 25 fps -> 5 Hz
WIN, HOP = 300, 150        # 60 s windows, 30 s hop (in 5 Hz steps)
HIDDEN, LAYERS, PROJ = 96, 2, 128
EPOCHS, BATCH, LR = 12, 32, 2e-3
DROPOUT = 0.2
NAN_INDICATOR = 0.05       # add a missing-flag column for features NaN on > 5% of samples
SEED = 0
CFG = {"DS": DS, "WIN": WIN, "HOP": HOP, "HIDDEN": HIDDEN, "LAYERS": LAYERS, "PROJ": PROJ,
       "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR, "DROPOUT": DROPOUT, "SEED": SEED}
MODEL_DIR = os.path.join(ev.ROOT, "models")


def _torch():
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch


def _halves(half):
    """[(start, stop)] of each half in a full-length array."""
    out = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


class Scaler:
    """Per-feature standardisation fitted on training samples, plus missing indicators."""

    def fit(self, X):
        X = np.asarray(X, np.float32)
        self.mean = np.nanmean(X, 0)
        self.std = np.nanstd(X, 0) + 1e-6
        self.mean = np.nan_to_num(self.mean)
        self.flag = np.flatnonzero(np.isnan(X).mean(0) > NAN_INDICATOR)
        return self

    def transform(self, X):
        X = np.asarray(X, np.float32)
        miss = np.isnan(X[:, self.flag]).astype(np.float32)
        Z = np.clip((X - self.mean) / self.std, -5, 5)
        return np.concatenate([np.nan_to_num(Z), miss], 1).astype(np.float32)

    def state(self):
        return {"mean": self.mean, "std": self.std, "flag": self.flag}

    @classmethod
    def from_state(cls, s):
        o = cls()
        o.mean, o.std, o.flag = s["mean"], s["std"], s["flag"]
        return o


def build_net(n_in):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(n_in, PROJ), nn.LayerNorm(PROJ), nn.GELU(),
                                      nn.Dropout(DROPOUT))
            self.gru = nn.GRU(PROJ, HIDDEN, num_layers=LAYERS, batch_first=True,
                              bidirectional=True, dropout=DROPOUT)
            self.head = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.proj(x))
            return self.head(h)
    return Net()


def _sequences(d, use_swap=True):
    """Downsampled (X, y, w) per half and orientation for one prepared match."""
    m = d["m"]
    w_full = ev.sample_weights(m["label"], ev.DEFAULT_CFG | {"weights": True})
    views = [(d["X"], m["label"])] + ([(d["Xs"], d["ys"])] if use_swap and d.get("Xs") is not None else [])
    out = []
    for a, b in _halves(m["half"]):
        idx = np.arange(a, b, DS)
        for X, y in views:
            out.append((np.asarray(X[idx], np.float32), y[idx].astype(np.int64), w_full[idx]))
    return out


def train_net(seqs, scaler, device=None, log=print):
    """Train on a list of (X, y, w) sequences (raw features; scaled here)."""
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    windows = []
    for X, y, w in seqs:
        Z = scaler.transform(X)
        for s in range(0, max(1, len(y) - WIN + 1), HOP):
            windows.append((Z[s:s + WIN], y[s:s + WIN], w[s:s + WIN]))
    Xw = torch.tensor(np.stack([np.pad(z, ((0, WIN - len(z)), (0, 0))) for z, _, _ in windows]))
    yw = torch.tensor(np.stack([np.pad(t, (0, WIN - len(t)), constant_values=C.X) for _, t, _ in windows]))
    ww = torch.tensor(np.stack([np.pad(v, (0, WIN - len(v))) for _, _, v in windows]), dtype=torch.float32)
    net = build_net(Xw.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(Xw) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(len(Xw), generator=g)
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            x, y, w = Xw[b].to(device), yw[b].to(device), ww[b].to(device)
            logits = net(x)
            mask = y != C.X
            loss = torch.nn.functional.cross_entropy(logits[mask], y[mask], reduction="none")
            loss = (loss * w[mask]).sum() / w[mask].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        log("  epoch {} loss {:.4f}".format(epoch + 1, total / len(Xw)))
    return net.cpu().eval()


def predict(net, scaler, X, half):
    """(T, 3) probabilities at 25 fps for a full-length feature matrix."""
    torch = _torch()
    T = len(half)
    out = np.full((T, 3), 1.0 / 3)
    with torch.no_grad():
        for a, b in _halves(half):
            idx = np.arange(a, b, DS)
            z = torch.tensor(scaler.transform(np.asarray(X[idx], np.float32)))[None]
            p = torch.softmax(net(z)[0], -1).numpy()
            frames = np.arange(a, b)
            for c in range(3):
                out[a:b, c] = np.interp(frames, idx, p[:, c])
    return out


def oof(data, log=print):
    """Leave-one-match-out network probabilities for prepared IDSSE `data`. Cached."""
    key = cache.digest("gru", sorted(CFG.items()), [d["key"] for d in data],
                       features.feature_key(features.build_features_v1))
    probs = []
    for k, held in enumerate(data):
        fk = cache.digest(key, k)
        if cache.exists("folds", fk):
            probs.append(cache.array("folds", fk, None, mmap=False))
            continue
        seqs = [s for j, d in enumerate(data) if j != k for s in _sequences(d)]
        scaler = Scaler().fit(np.concatenate([x for x, _, _ in seqs])[::7])
        log("fold {}: {} sequences".format(held["m"]["name"], len(seqs)))
        net = train_net(seqs, scaler, log=log)
        p = predict(net, scaler, held["X"], held["m"]["half"])
        probs.append(cache.array("folds", fk, lambda: p, mmap=False))
    return probs


def blend(p_gbm, p_nn, alpha=0.5):
    """Geometric mean of the two models' probabilities, renormalised."""
    lp = alpha * np.log(np.clip(p_gbm, 1e-6, 1)) + (1 - alpha) * np.log(np.clip(p_nn, 1e-6, 1))
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def train_final(data, name, log=print):
    """Network on all prepared matches; weights + scaler saved to models/<name>.pt."""
    torch = _torch()
    seqs = [s for d in data for s in _sequences(d)]
    scaler = Scaler().fit(np.concatenate([x for x, _, _ in seqs])[::7])
    net = train_net(seqs, scaler, log=log)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name + ".pt")
    torch.save({"state": net.state_dict(), "scaler": scaler.state(), "cfg": CFG,
                "n_in": int(sum(p.numel() for p in []) or net.proj[0].in_features)}, path)
    return path


def load_final(path):
    torch = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(blob["n_in"])
    net.load_state_dict(blob["state"])
    return net.eval(), Scaler.from_state(blob["scaler"])
