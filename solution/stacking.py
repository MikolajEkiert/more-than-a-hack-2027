"""Stage 2: stacked sequential learning on the stage-1 probabilities.

Cohen & Carvalho (IJCAI 2005); multi-scale version Puertas, Escalera & Pujol (2009). A second
classifier sees the first one's predictions over windows of 0.2-10 s, before, after and around
each frame, so it can learn temporal structure that per-frame features miss -- e.g. "a brief A
blip inside a long H run is a touch, not a possession". This is the same role as the refinement
stages of MS-TCN-style action segmentation, at a fraction of the cost.

Trained on OUT-OF-FOLD stage-1 probabilities (each match predicted by a model that never saw it),
so stage 2 learns from the same kind of imperfect inputs it gets on BAR-ATH.
"""

import numpy as np

from solution import container as C, evaluate_external as ev, features

# Params
CENTRED = (5, 25, 75, 250)        # frames: 0.2 s, 1 s, 3 s, 10 s
SIDE = (25, 100, 250)             # past-only / future-only windows
ROUNDS = 200
STRIDE = 5


def _swap_labels(lab):
    out = lab.copy()
    out[lab == C.H], out[lab == C.A] = C.A, C.H
    return out


def stage2_features(prob, half):
    """(T, K) multi-scale features of stage-1 probabilities; frames outside play are masked."""
    p = np.asarray(prob, np.float64).copy()
    p[half == 0] = np.nan
    series = {"pH": p[:, 0], "pA": p[:, 1], "pD": p[:, 2], "dHA": p[:, 0] - p[:, 1]}
    cols, names = [], []
    for key, x in series.items():
        cols.append(x)
        names.append(key)
        for w in CENTRED:
            cols.append(features._roll(x, -w, w))
            names.append("{}_c{}".format(key, w))
        for w in SIDE:
            cols.append(features._roll(x, -w, -1))
            names.append("{}_p{}".format(key, w))
            cols.append(features._roll(x, 1, w))
            names.append("{}_f{}".format(key, w))
    return np.column_stack(cols).astype(np.float32), names


def train_stage2(probs, labels, halves):
    """One stage-2 model from several matches' (out-of-fold) probabilities, both orientations."""
    X, y = [], []
    for p, lab, half in zip(probs, labels, halves):
        for pp, ll in ((p, lab), (np.asarray(p)[:, [1, 0, 2]], _swap_labels(lab))):
            F, _ = stage2_features(pp, half)
            rows = np.flatnonzero(ll != C.X)[::STRIDE]
            X.append(F[rows])
            y.append(ll[rows])
    return ev.train(np.concatenate(X), np.concatenate(y), rounds=ROUNDS)


def lomo(res, bootstrap=200):
    """Leave-one-match-out stage 2 on top of an ev.run() result, scored by ev.score()."""
    data, probs = res["data"], res["probs"]
    labels = [d["m"]["label"] for d in data]
    halves = [d["m"]["half"] for d in data]
    out = []
    for k in range(len(data)):
        others = [j for j in range(len(data)) if j != k]
        model = train_stage2([probs[j] for j in others], [labels[j] for j in others],
                             [halves[j] for j in others])
        out.append(model.predict(stage2_features(probs[k], halves[k])[0]))
    return ev.score(out, data, bootstrap=bootstrap)


def apply(model, prob, half):
    return model.predict(stage2_features(prob, half)[0])
