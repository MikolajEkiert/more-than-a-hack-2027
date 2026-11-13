"""Boundary-aware decoder vs the current decoder, same folds, IDSSE only.

    python -m solution.experiments_boundary

beta = 0 configurations ARE the current decoder, so restricting the nested penalty choice to them
reproduces today's number on identical out-of-fold probabilities; letting it use every
configuration measures what the boundary model adds.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import boundary, container as C, evaluate_external as ev, features, labels  # noqa: E402

pm = C.pm
BINS = [0, 1, 2, 5, 15, 1e9]
BIN_NAMES = ["<1s", "1-2s", "2-5s", "5-15s", ">15s"]


def nested(grid, allowed):
    """Per held-out match: best allowed config on the other matches -> its score there."""
    out, picks = [], []
    for k in range(grid.shape[1]):
        others = [j for j in range(grid.shape[1]) if j != k]
        g = allowed[int(np.argmax(grid[allowed][:, others].mean(1)))]
        out.append(grid[g, k])
        picks.append(g)
    return np.array(out), picks


def recall_by_length(data, preds):
    n, hit = np.zeros(5), np.zeros(5)
    for d, pred in zip(data, preds):
        gt = d["m"]["label"]
        runs = pm.valid_runs(gt)
        G, P = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
        used_g, used_p = set(), set()
        for iou, i, j in pm.spell_overlaps(G, P):
            if iou < 0.5:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
        for i, (a, b, _) in enumerate(G):
            k = np.digitize((b - a) / C.FPS, BINS) - 1
            n[k] += 1
            hit[k] += i in used_g
    return hit / np.maximum(n, 1)


def main():
    t0 = time.time()
    path = os.path.join(ROOT, "data", "models", "degrade_params.json")
    params = json.load(open(path)) if os.path.isfile(path) else None
    res = boundary.lomo(features.build_features_v1, labels.raw, params, bootstrap=0,
                        log=lambda s: None)
    grid, configs, data = res["grid"], res["pairs"], res["data"]
    current = [i for i, c in enumerate(configs) if c[1] == 0]
    every = list(range(len(configs)))
    s_cur, p_cur = nested(grid, current)
    s_new, p_new = nested(grid, every)

    # decoded predictions for both choices, to break recall down by possession length
    def preds_for(picks):
        return [boundary.decode_with(res["probs"][k], d["m"]["half"], res["p_boundary"][k],
                                     [configs[picks[k]]])[0] for k, d in enumerate(data)]
    pr_cur, pr_new = preds_for(p_cur), preds_for(p_new)
    sep = np.full(C.HALF_GAP, C.X, np.int8)
    gt_all = np.concatenate([x for d in data for x in (d["m"]["label"], sep)])
    pooled = {}
    for name, preds in (("current", pr_cur), ("boundary", pr_new)):
        pred_all = np.concatenate([x for p in preds for x in (p, sep)])
        pooled[name] = pm.evaluate(gt_all, pred_all, bootstrap_n=200)

    print("[{:.0f}s]".format(time.time() - t0))
    print("{:<10} {:>8} {:>18} {:>8} {:>7}   recall by possession length ({})".format(
        "decoder", "spellF1", "interval", "macroF1", "poss", " / ".join(BIN_NAMES)))
    for name, preds in (("current", pr_cur), ("boundary", pr_new)):
        p = pooled[name]
        rec = recall_by_length(data, preds)
        print("{:<10} {:>8.4f} {:>8.4f} .. {:.4f} {:>8.4f} {:>7d}   {}".format(
            name, p["spell_f1_mean"], p["spell_bootstrap_low"], p["spell_bootstrap_high"],
            p["macro_f1"], p["n_pred_spells"], " / ".join("{:.0%}".format(r) for r in rec)))
    print("per match, boundary - current:", " ".join("{:+.3f}".format(x) for x in s_new - s_cur),
          "| wins {}/{}".format(int(np.sum(s_new > s_cur)), len(s_cur)))
    best = int(np.argmax(grid.mean(1)))
    print("most chosen config (lambda, beta, dead):", configs[max(set(p_new), key=p_new.count)],
          "| best on average:", configs[best])
    json.dump({"best_config": configs[best], "pooled_current": pooled["current"]["spell_f1_mean"],
               "pooled_boundary": pooled["boundary"]["spell_f1_mean"]},
              open(os.path.join(ROOT, "data", "external", "idsse", "oof", "boundary.json"), "w"))


if __name__ == "__main__":
    main()
