"""Boundary-aware decoding (after ASRF, Ishikawa et al., WACV 2021).

The plain decoder charges the same penalty for a team switch on every frame, so a 2-5 s
possession rarely pays for its two switches -- our largest error (recall 48% in that band on
IDSSE). Here a second LightGBM predicts "possession changes hands within +/-TOL frames of t" from
the same features (touches, nearest-team streaks, ...), and the decoder's team-switch cost becomes

    cost_t = max(FLOOR, lam - beta * logit(p_boundary_t))

cheap where a hand-over is likely, dearer than today where it is not. beta = 0 is the current
decoder, so one grid compares both on identical folds.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d

from solution import cache, container as C, decode, evaluate_external as ev, features, labels

# Params
TOL = 3                      # frames either side of a hand-over counted as "boundary"
NEG_STRIDE = 5               # keep every positive, every 5th negative
POOL = 5                     # max-pool the boundary probability over 5 frames
FLOOR = 0.5
LAMBDAS = (4, 8, 16, 32)
BETAS = (0, 1, 2, 4)
DEADS = (16, 32, 64)
CONFIGS = [(lam, beta, dead) for lam in LAMBDAS for beta in BETAS for dead in DEADS]
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbose": -1, "seed": 0, "deterministic": True, "force_row_wise": True,
}
ROUNDS = 300


def targets(label):
    """1 near a direct H<->A hand-over, 0 elsewhere, -1 where the label is X."""
    T = len(label)
    team = np.isin(label, (C.H, C.A))
    idx = np.flatnonzero((label[1:] != label[:-1]) & team[1:] & team[:-1]) + 1
    b = np.zeros(T, np.int8)
    for o in range(-TOL, TOL + 1):
        b[np.clip(idx + o, 0, T - 1)] = 1
    b[label == C.X] = -1
    return b


def _rows(b, neg_stride=NEG_STRIDE):
    pos = np.flatnonzero(b == 1)
    neg = np.flatnonzero(b == 0)[::neg_stride]
    return np.sort(np.concatenate([pos, neg]))


def train_boundary(data, skip=None, neg_stride=NEG_STRIDE):
    skip = set() if skip is None else ({skip} if isinstance(skip, (int, np.integer)) else set(skip))
    X, y = [], []
    for j, d in enumerate(data):
        if j in skip:
            continue
        b = targets(d["m"]["label"])          # swapping teams leaves hand-overs where they are
        rows = _rows(b, neg_stride)
        for F in (d["X"], d.get("Xs")):
            if F is None:
                continue
            X.append(F[rows])
            y.append(b[rows])
    return ev.train(np.concatenate(X), np.concatenate(y), params=PARAMS, rounds=ROUNDS)


def team_costs(p_boundary, configs=CONFIGS):
    p = np.clip(maximum_filter1d(np.asarray(p_boundary, np.float64), POOL), 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))
    lam = np.array([c[0] for c in configs], np.float64)[:, None]
    beta = np.array([c[1] for c in configs], np.float64)[:, None]
    return np.maximum(FLOOR, lam - beta * logit[None, :])


def decode_with(prob, half, p_boundary, configs=CONFIGS):
    return decode.decode_tv(prob, half, team_costs(p_boundary, configs),
                            np.array([c[2] for c in configs], np.float64))


def evaluate(matches=None, feature_fn=features.build_features_v1, label_fn=labels.raw,
             degrade_params=None, seed=0, n_folds=None, stride=ev.TRAIN_STRIDE, swap=True,
             light=False, neg_stride=NEG_STRIDE, bootstrap=200, log=print, train_cfg=None):
    """Stage-1 out-of-fold probs (ev.run, cached) + boundary model per fold (cached), scored over
    CONFIGS. Defaults reproduce the IDSSE leave-one-match-out setup; pass PFF loaders + n_folds.
    train_cfg is the stage-1 variant (the final pipeline uses {"weights": True})."""
    kw = dict(feature_fn=feature_fn, label_fn=label_fn, degrade_params=degrade_params, seed=seed,
              log=log, swap=swap, light=light)
    base = ev.run(matches=matches, n_folds=n_folds, stride=stride, bootstrap=0, train_cfg=train_cfg, **kw)
    data = ev.prepare(matches if matches is not None else ev.idsse.load_all(), **kw)
    pb = oof(data, feature_fn, n_folds, swap, neg_stride, log)
    for d in data:
        d.pop("X")
        d.pop("Xs", None)
    res = ev.score(base["probs"], data, pairs=CONFIGS, bootstrap=bootstrap,
                   decode_fn=lambda k, prob, half: decode_with(prob, half, pb[k]))
    res["p_boundary"] = pb
    res["base"] = base
    return res


def oof(data, feature_fn, n_folds=None, swap=True, neg_stride=NEG_STRIDE, log=print):
    """Out-of-fold boundary probabilities for prepared `data` (features still attached). Cached."""
    n = len(data)
    fold_of = [i % n_folds for i in range(n)] if n_folds else list(range(n))
    default = n_folds is None and swap and neg_stride == NEG_STRIDE
    fkey = features.feature_key(feature_fn)
    run_key = (cache.digest("boundary", fkey, [d["key"] for d in data], PARAMS, ROUNDS, TOL, NEG_STRIDE)
               if default else
               cache.digest("boundary", fkey, [d["key"] for d in data], PARAMS, ROUNDS, TOL,
                            neg_stride, swap, fold_of))
    key_of = (lambda f, i: cache.digest(run_key, f)) if default else \
             (lambda f, i: cache.digest(run_key, f, i))
    pb = [None] * n
    for f in sorted(set(fold_of)):
        members = [i for i in range(n) if fold_of[i] == f]
        if all(cache.exists("folds", key_of(f, i)) for i in members):
            for i in members:
                pb[i] = cache.array("folds", key_of(f, i), None, mmap=False)
            continue
        model = train_boundary(data, skip=members, neg_stride=neg_stride)
        for i in members:
            pb[i] = cache.array("folds", key_of(f, i), lambda i=i: model.predict(data[i]["X"]),
                                mmap=False)
        log("boundary fold {} done".format(f))
    return pb


def lomo(feature_fn, label_fn, degrade_params, seed=0, bootstrap=200, log=print):
    """The IDSSE leave-one-match-out setup (kept for experiments_boundary.py)."""
    return evaluate(None, feature_fn, label_fn, degrade_params, seed, bootstrap=bootstrap, log=log)


def train_final(feature_fn, label_fn, degrade_params, seed=0, extra=None, extra_neg_stride=15):
    """Boundary model on all IDSSE matches (+ `extra` loaders, e.g. PFF), cached."""
    full = sorted(dict(ev.degrade.DEFAULTS, **(degrade_params or {})).items())
    extra = extra or []
    key = cache.digest("boundary-final", features.feature_key(feature_fn),
                       labels.label_key(label_fn), full, seed, cache.source(ev.degrade),
                       PARAMS, ROUNDS, TOL, NEG_STRIDE, len(extra), extra_neg_stride)

    def fit():
        data = ev.prepare(ev.idsse.load_all(), feature_fn, label_fn, seed=seed,
                          degrade_params=degrade_params, log=lambda s: None)
        if extra:
            more = ev.prepare(extra, feature_fn, label_fn, seed=seed, degrade_params=degrade_params,
                              log=lambda s: None, swap=False, light=True)
            m1 = _boundary_matrix(data, NEG_STRIDE)
            m2 = _boundary_matrix(more, extra_neg_stride)
            return ev.train(np.concatenate([m1[0], m2[0]]), np.concatenate([m1[1], m2[1]]),
                            params=PARAMS, rounds=ROUNDS)
        return train_boundary(data)
    return cache.model(key, fit)


def _boundary_matrix(data, neg_stride):
    X, y = [], []
    for d in data:
        b = targets(d["m"]["label"])
        rows = _rows(b, neg_stride)
        for F in (d["X"], d.get("Xs")):
            if F is not None:
                X.append(F[rows])
                y.append(b[rows])
    return np.concatenate(X), np.concatenate(y)
