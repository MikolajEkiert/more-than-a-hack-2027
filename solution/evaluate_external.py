"""The evaluation we trust: leave-one-match-out on IDSSE, ATD-like noise, the official scorer.

    python -m solution.evaluate_external                 # all 7 matches, degraded (the real number)
    python -m solution.evaluate_external --clean         # no degradation (upper bound, for contrast)
    python -m solution.evaluate_external --matches J03WMX J03WN1 J03WOH   # quicker, 3 folds

    from solution import evaluate_external as ev        # from the notebook, with your own features
    res = ev.run(feature_fn=my_features)

For every held-out match:
  1. degrade it and the six training matches to look like the BAR-ATH ATD   (solution/degrade.py)
  2. build the same features BAR-ATH will get                                (feature_fn)
  3. train LightGBM on the six (plus home/away-swapped copies), predict the held-out match
  4. decode with Viterbi over a grid of switch penalties                    (solution/decode.py)
  5. score with evaluation/possession_metrics.py -- the organisers' exact metric code

The penalty pair reported for a match is chosen on the OTHER six matches only (nested), so the
per-match numbers are honest out-of-sample estimates. The organisers' baseline (nearest player,
majority vote +/-50 frames) runs on the same degraded data as a yardstick: it scores 0.2433 on the
real BAR-ATH match, so landing near that here says the degradation is realistic.

The public BAR-ATH window is deliberately not used anywhere in this file.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from solution import cache, container as C, decode, degrade, features, idsse, labels  # noqa: E402

pm = C.pm

# Params
TEAM_PENALTIES = (0.5, 1, 2, 4, 8, 16)
DEAD_PENALTIES = (8, 16, 32, 64, 128)
PAIRS = [(t, d) for t in TEAM_PENALTIES for d in DEAD_PENALTIES]
LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 0,
    "deterministic": True, "force_row_wise": True,   # the rules demand a byte-identical rerun
}
ROUNDS = 300
TRAIN_STRIDE = 5          # every 5th frame: neighbours are near-duplicates, 5x less memory
BASELINE_WINDOW = 50      # the published floor, baseline_smooth50
OUT_DIR = os.path.join(ROOT, "data", "external", "idsse", "oof")


def baseline_codes(m, window=BASELINE_WINDOW):
    """The organisers' baseline on a container: nearest tracked player, then majority vote."""
    import baseline  # starter/baseline.py
    dist = np.hypot(m["px"] - m["bx"][:, None], m["py"] - m["by"][:, None])
    ok = ~np.all(np.isnan(dist), axis=1)
    codes = np.full(len(dist), baseline.NOBODY, np.int8)
    codes[ok] = m["team"][np.nanargmin(dist[ok], axis=1)]
    return pm.label_codes(baseline.smooth(codes, window))


def spell_f1(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def train(X, y, params=None, rounds=ROUNDS, weight=None):
    import lightgbm as lgb
    return lgb.train(params or LGB_PARAMS, lgb.Dataset(X, y, weight=weight, free_raw_data=True), rounds)


# -- training variants (competitor review, research/COMPETITORS.md) ----------------------------
# weights:    frames near any label change and inside short possessions count more in the loss
#             (boundary-weighted CE in Sigue / SAM / Mantas) -- spellF1 is decided there.
# factorized: P(dead) and P(home | alive) as two binary models (Fable, Kora, most mature repos);
#             combined as P(H) = (1 - P(D)) * P(H | alive).
DEFAULT_CFG = {"weights": False, "factorized": False, "w_boundary": 5.0, "w_short": 3.0,
               "near": 10, "short_s": 5.0}
BIN_PARAMS = dict(LGB_PARAMS, objective="binary")
BIN_PARAMS.pop("num_class")


def sample_weights(label, cfg):
    """1 everywhere, + w_boundary within `near` frames of any change, + w_short in short spells."""
    w = np.ones(len(label))
    ok = label != C.X
    chg = np.flatnonzero((label[1:] != label[:-1]) & ok[1:] & ok[:-1]) + 1
    near = np.zeros(len(label), bool)
    for o in range(-cfg["near"], cfg["near"] + 1):
        near[np.clip(chg + o, 0, len(label) - 1)] = True
    w[near] += cfg["w_boundary"]
    for a, b, _ in pm.possession_spells(label, pm.valid_runs(label)):
        if b - a < cfg["short_s"] * C.FPS:
            w[a:b] += cfg["w_short"]
    return w


class TwoHead:
    """P(dead) x P(home | alive) behind the same predict() as a multiclass booster."""

    def __init__(self, dead, team):
        self.dead, self.team = dead, team

    def predict(self, X):
        pd, ph = self.dead.predict(X), self.team.predict(X)
        return np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd])

    def feature_importance(self, kind="gain"):
        return self.dead.feature_importance(kind) + self.team.feature_importance(kind)


def fit(data, skip=None, stride=TRAIN_STRIDE, cfg=None, extra=None):
    """Train the stage-1 model of a variant. `extra` = (X, y, w) appended (another dataset)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    X, y = stack(data, skip=skip, stride=stride)
    w = stack_weights(data, skip, stride, cfg) if cfg["weights"] else None
    if extra is not None:
        X, y = np.concatenate([X, extra[0]]), np.concatenate([y, extra[1]])
        if w is not None:
            w = np.concatenate([w, extra[2] if extra[2] is not None else np.ones(len(extra[1]))])
    if not cfg["factorized"]:
        return train(X, y, weight=w)
    dead = train(X, (y == C.D).astype(np.int8), params=BIN_PARAMS, weight=w)
    alive = y != C.D
    team = train(X[alive], (y[alive] == C.H).astype(np.int8), params=BIN_PARAMS,
                 weight=None if w is None else w[alive])
    return TwoHead(dead, team)


def stack_weights(data, skip=None, stride=TRAIN_STRIDE, cfg=None):
    """Sample weights aligned with stack(data, skip, stride)."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    skip = set() if skip is None else ({skip} if isinstance(skip, (int, np.integer)) else set(skip))
    out = []
    for j, d in enumerate(data):
        if j in skip:
            continue
        w = sample_weights(d["m"]["label"], cfg)
        views = [d["m"]["label"]] + ([d["ys"]] if d.get("Xs") is not None else [])
        for y in views:
            out.append(w[np.flatnonzero(y != C.X)[::stride]])
    return np.concatenate(out)


def _top_up(raw, params):
    """Degradation params for one match, counting ball frames that are already missing.

    IDSSE has the ball on every frame, so nothing changes for it (and its cache keys stay). PFF's
    broadcast tracking already loses the ball sometimes; removing 41% on top would overshoot.
    """
    play = raw["half"] > 0
    existing = float(np.mean(np.isnan(raw["bx"][play]))) if play.any() else 0.0
    if existing < 1e-3:
        return params
    target = dict(degrade.DEFAULTS, **(params or {}))["ball_missing"]
    return dict(params or {}, ball_missing=max(0.0, (target - existing) / (1 - existing)))


def prepare(matches, feature_fn=features.build_features, label_fn=labels.raw,
            clean=False, seed=0, degrade_params=None, log=print, swap=True, light=False):
    """Degrade, label and featurise each match, original and home/away-swapped. Cached on disk.

    `matches` items may be containers or zero-argument loaders (PFF: 64 matches do not fit in
    RAM). light=True keeps only labels / halves / baseline per match -- enough for scoring.
    """
    out, t0 = [], time.time()
    lkey = labels.label_key(label_fn)
    for i, item in enumerate(matches):
        raw = item() if callable(item) else item
        if clean:
            m, mkey = raw, cache.digest("clean", cache.fingerprint(raw))
        else:
            dseed = seed * 100 + i
            params_i = _top_up(raw, degrade_params)
            mkey = degrade.degrade_key(raw, dseed, params_i)
            m = cache.container("degraded", mkey, lambda: degrade.degrade(
                raw, seed=dseed, **(params_i or {})))
        lab = cache.array("labels", cache.digest(lkey, cache.fingerprint(raw)),
                          lambda: label_fn(raw), mmap=False)
        m = dict(m, label=np.asarray(lab, np.int8))
        Xo, names = features.cached(feature_fn, m, "orig", mkey)
        entry = {"m": m, "X": Xo, "Xs": None, "ys": None, "names": names,
                 "key": cache.digest(mkey, lkey)}
        if swap:
            s = C.swap_teams(m)
            entry["Xs"], _ = features.cached(feature_fn, s, "swap", mkey)
            entry["ys"] = s["label"]
        if light:
            entry["base"] = baseline_codes(m)
            entry["m"] = {k: m[k] for k in ("label", "half", "name", "team")}
        out.append(entry)
        log("[{:5.0f}s] features {} {}".format(time.time() - t0, m["name"], Xo.shape))
    return out


def stack(data, skip=None, stride=TRAIN_STRIDE):
    """Training matrix from every prepared match except those in `skip`, both orientations."""
    skip = set() if skip is None else ({skip} if isinstance(skip, (int, np.integer)) else set(skip))
    Xtr, ytr = [], []
    for j, d in enumerate(data):
        if j in skip:
            continue
        views = [(d["X"], d["m"]["label"])]
        if d.get("Xs") is not None:
            views.append((d["Xs"], d["ys"]))
        for X, y in views:
            rows = np.flatnonzero(y != C.X)[::stride]
            Xtr.append(X[rows])
            ytr.append(y[rows])
    return np.concatenate(Xtr), np.concatenate(ytr)


def run(match_ids=None, feature_fn=features.build_features, label_fn=labels.raw, clean=False,
        seed=0, degrade_params=None, bootstrap=200, pairs=PAIRS, log=print,
        matches=None, n_folds=None, stride=TRAIN_STRIDE, swap=True, light=False, train_cfg=None):
    """Out-of-fold evaluation. Returns a dict with per-match rows, pooled metrics, grid, gains.

    Default: the seven IDSSE matches, leave-one-match-out. `matches` (containers or loaders)
    swaps in another dataset; `n_folds` groups matches into k folds (64 PFF matches); `stride`
    and `swap` trade training size for memory.

    Every slow step is cached in data/cache/ (see solution/cache.py): a rerun with nothing
    changed takes about a minute (decoding + scoring), a new feature block recomputes only its
    own feature set and folds.
    """
    t0 = time.time()
    data = prepare(matches if matches is not None else idsse.load_all(match_ids=match_ids),
                   feature_fn, label_fn, clean, seed, degrade_params, log, swap=swap, light=light)
    names = data[0]["names"]
    n = len(data)
    fold_of = [i % n_folds for i in range(n)] if n_folds else list(range(n))

    # one cached prediction per held-out match; the default IDSSE setup keeps its original keys
    cfg = dict(DEFAULT_CFG, **(train_cfg or {}))
    default = n_folds is None and stride == TRAIN_STRIDE and swap and cfg == DEFAULT_CFG
    fkey = features.feature_key(feature_fn)
    run_key = (cache.digest("fold", fkey, [d["key"] for d in data], LGB_PARAMS, ROUNDS, TRAIN_STRIDE)
               if default else
               cache.digest("fold", fkey, [d["key"] for d in data], LGB_PARAMS, ROUNDS, stride,
                            swap, fold_of, sorted(cfg.items())))
    prob_key = (lambda f, i: cache.digest(run_key, f)) if default else \
               (lambda f, i: cache.digest(run_key, f, i))
    probs, gains = [None] * n, []
    for f in sorted(set(fold_of)):
        members = [i for i in range(n) if fold_of[i] == f]
        gain_key = cache.digest(cache.digest(run_key, f), "gain")
        if all(cache.exists("folds", prob_key(f, i)) for i in members) and cache.exists("folds", gain_key):
            for i in members:
                probs[i] = cache.array("folds", prob_key(f, i), None, mmap=False)
            gains.append(cache.array("folds", gain_key, None, mmap=False))
            log("[{:5.0f}s] fold {} from cache".format(time.time() - t0, f))
            continue
        model = fit(data, skip=members, stride=stride, cfg=cfg)
        for i in members:
            prob = model.predict(data[i]["X"])
            probs[i] = cache.array("folds", prob_key(f, i), lambda: prob, mmap=False)
        gain = model.feature_importance("gain")
        gains.append(cache.array("folds", gain_key, lambda: gain, mmap=False))
        log("[{:5.0f}s] fold {} ({} held out) trained".format(time.time() - t0, f, len(members)))

    for d in data:  # features are memory-mapped; the rest of the run needs labels and probs only
        d.pop("X")
        d.pop("Xs")

    res = score(probs, data, pairs, bootstrap)
    res.update(names=names, gain=np.mean(gains, axis=0), seconds=time.time() - t0)
    return res


def _weighted(data, skip, stride, cfg):
    """(X, y, w) for a training set; w is None unless cfg asks for sample weights."""
    X, y = stack(data, skip=skip, stride=stride)
    return X, y, (stack_weights(data, skip, stride, cfg) if cfg["weights"] else None)


def oof_with_extra(data, extra, n_folds=None, stride=TRAIN_STRIDE, extra_stride=15, log=print,
                   train_cfg=None):
    """Out-of-fold probabilities for `data` where every fold ALSO trains on all of `extra`.

    "Train on both datasets, test on one": the held-out matches of `data` are never seen, `extra`
    (the other dataset) is always in. Score the result with score(probs, data).
    """
    cfg = dict(DEFAULT_CFG, **(train_cfg or {}))
    n = len(data)
    fold_of = [i % n_folds for i in range(n)] if n_folds else list(range(n))
    parts = ["oofx", [d["key"] for d in data], [d["key"] for d in extra], LGB_PARAMS,
             ROUNDS, stride, extra_stride, fold_of]
    key = cache.digest(*(parts if cfg == DEFAULT_CFG else parts + [sorted(cfg.items())]))
    Xe, ye, we = _weighted(extra, None, extra_stride, cfg)
    probs = [None] * n
    for f in sorted(set(fold_of)):
        members = [i for i in range(n) if fold_of[i] == f]
        keys = {i: cache.digest(key, f, i) for i in members}
        if all(cache.exists("folds", keys[i]) for i in members):
            for i in members:
                probs[i] = cache.array("folds", keys[i], None, mmap=False)
            continue
        X, y, w = _weighted(data, members, stride, cfg)
        model = train(np.concatenate([X, Xe]), np.concatenate([y, ye]),
                      weight=None if w is None else np.concatenate([w, we]))
        del X
        for i in members:
            probs[i] = cache.array("folds", keys[i], lambda i=i: model.predict(data[i]["X"]), mmap=False)
        log("fold {} (+{} extra rows) done".format(f, len(ye)))
    return probs


def cross(train_data, test_data, stride=TRAIN_STRIDE, pairs=PAIRS, bootstrap=200, log=print,
          train_cfg=None):
    """Train one model on every match of `train_data`, score every match of `test_data`.

    Both are prepare() outputs that still hold their features. Used for the dataset matrix
    (IDSSE -> PFF, PFF -> IDSSE): does what is learned on one definition transfer to another?
    """
    cfg = dict(DEFAULT_CFG, **(train_cfg or {}))
    parts = ["cross", [d["key"] for d in train_data], LGB_PARAMS, ROUNDS, stride]
    key = cache.digest(*(parts if cfg == DEFAULT_CFG else parts + [sorted(cfg.items())]))

    def fit():
        X, y, w = _weighted(train_data, None, stride, cfg)
        return train(X, y, weight=w)
    model = cache.model(key, fit)
    probs = [cache.array("folds", cache.digest(key, d["key"]), lambda d=d: model.predict(d["X"]),
                         mmap=False) for d in test_data]
    log("cross: trained on {} matches, scored {}".format(len(train_data), len(test_data)))
    return score(probs, test_data, pairs, bootstrap)


def score(probs, data, pairs=PAIRS, bootstrap=200, decode_fn=None):
    """Decode out-of-fold probabilities, choose penalties nested, score with the official code.

    Shared by run() and by anything producing its own per-match probabilities (stacking, ...),
    so every variant is judged by exactly the same procedure. decode_fn(k, prob, half) -> (G, T)
    replaces the default Viterbi; `pairs` then just names its G configurations.
    """
    t0 = time.time()
    grid = np.zeros((len(pairs), len(data)))
    decoded = []
    for k, d in enumerate(data):
        paths = (decode_fn(k, probs[k], d["m"]["half"]) if decode_fn
                 else decode.decode(probs[k], d["m"]["half"], pairs))
        decoded.append(paths)
        grid[:, k] = [spell_f1(d["m"]["label"], p) for p in paths]

    rows, preds, pooled_gt, pooled_pred, pooled_base = [], [], [], [], []
    sep = np.full(C.HALF_GAP, C.X, np.int8)
    for k, d in enumerate(data):
        others = [j for j in range(len(data)) if j != k]
        g = int(np.argmax(grid[:, others].mean(1))) if others else int(np.argmax(grid[:, k]))
        gt, pred = d["m"]["label"], decoded[k][g]
        base = d["base"] if "base" in d else baseline_codes(d["m"])
        preds.append(pred)
        r, rb = pm.evaluate(gt, pred), pm.evaluate(gt, base)
        rows.append({
            "match": d["m"]["name"], "pair": tuple(pairs[g]),
            "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
            "base_spellF1": rb["spell_f1_mean"], "base_macroF1": rb["macro_f1"],
            "base_n_pred": rb["n_pred_spells"],
        })
        for lst, arr in ((pooled_gt, gt), (pooled_pred, pred), (pooled_base, base)):
            lst.extend([arr, sep])

    gt_all, pred_all, base_all = (np.concatenate(x) for x in (pooled_gt, pooled_pred, pooled_base))
    pooled = pm.evaluate(gt_all, pred_all, bootstrap_n=bootstrap)
    pooled_b = pm.evaluate(gt_all, base_all)
    best = int(np.argmax(grid.mean(1)))
    return {"rows": rows, "pooled": pooled, "pooled_baseline": pooled_b, "grid": grid,
            "pairs": pairs, "best_pair": tuple(pairs[best]), "probs": probs, "preds": preds,
            "data": data, "names": [], "gain": np.zeros(0), "seconds": time.time() - t0}


def report(res, tag=""):
    p, pb = res["pooled"], res["pooled_baseline"]
    print("\n{} | {} matches | {:.0f}s".format(tag.upper(), len(res["rows"]), res["seconds"]))
    fmt = "{:<20} {:>10} {:>8.4f} {:>8.4f} {:>6d} {:>6d}  {:>9.4f} {:>9.4f} {:>6d}"
    print("{:<20} {:>10} {:>8} {:>8} {:>6} {:>6}  {:>9} {:>9} {:>6}".format(
        "held-out match", "penalties", "spellF1", "macroF1", "gt", "pred",
        "base sF1", "base mF1", "pred"))
    for r in res["rows"]:
        print(fmt.format(r["match"], "{}/{}".format(*r["pair"]), r["spellF1"], r["macroF1"],
                         r["n_gt"], r["n_pred"], r["base_spellF1"], r["base_macroF1"],
                         r["base_n_pred"]))
    print(fmt.format("POOLED", "", p["spell_f1_mean"], p["macro_f1"], p["n_gt_spells"],
                     p["n_pred_spells"], pb["spell_f1_mean"], pb["macro_f1"], pb["n_pred_spells"]))
    if p.get("spell_bootstrap_low") is not None:
        print("pooled spellF1 {}-{} bootstrap interval: {:.4f} .. {:.4f}".format(
            *pm.BOOTSTRAP_PERCENTILES, p["spell_bootstrap_low"], p["spell_bootstrap_high"]))
    grid, pairs = res["grid"], res["pairs"]
    print("\nbest penalty pairs by mean spellF1 over matches (team/dead):")
    for g in np.argsort(-grid.mean(1))[:5]:
        print("  {:>4}/{:<4} {:.4f}".format(pairs[g][0], pairs[g][1], grid[g].mean()))
    if len(res["gain"]):
        top = np.argsort(-res["gain"])[:20]
        print("\ntop features by gain:", ", ".join(res["names"][i] for i in top))


def save(res, tag):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "{}.npz".format(tag)),
                        **{d["m"]["name"]: res["probs"][k] for k, d in enumerate(res["data"])})
    with open(os.path.join(OUT_DIR, "{}.json".format(tag)), "w") as handle:
        json.dump({"rows": res["rows"], "pooled_spellF1": res["pooled"]["spell_f1_mean"],
                   "pooled_macroF1": res["pooled"]["macro_f1"], "pairs": res["pairs"],
                   "best_pair": res["best_pair"], "grid": res["grid"].tolist(),
                   "features": res["names"]}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=sorted(idsse.MATCHES))
    parser.add_argument("--clean", action="store_true", help="skip the ATD-like degradation")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="merge sandwiched touches shorter than this many seconds (0 = raw)")
    parser.add_argument("--features", default="v0", choices=sorted(features.FEATURE_SETS))
    parser.add_argument("--degrade-json", default=None,
                        help="calibrated degradation parameters (the notebook writes "
                             "data/models/degrade_params.json)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=200, help="draws for the pooled CI")
    parser.add_argument("--tag", default=None, help="name for the saved OOF predictions")
    args = parser.parse_args()
    tag = args.tag or "{}_{}".format(args.features, "clean" if args.clean else "degraded") + (
        "_tau{}".format(args.tau) if args.tau else "")
    label_fn = labels.harmonised(args.tau) if args.tau else labels.raw
    params = json.load(open(args.degrade_json)) if args.degrade_json else None
    res = run(args.matches, feature_fn=features.FEATURE_SETS[args.features], label_fn=label_fn,
              clean=args.clean, seed=args.seed, degrade_params=params, bootstrap=args.bootstrap)
    report(res, tag)
    save(res, tag)


if __name__ == "__main__":
    main()
