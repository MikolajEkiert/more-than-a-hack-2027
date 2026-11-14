"""Compare improvement candidates on IDSSE (leave-one-match-out). The public window is not used.

    python -m solution.experiments

Variants, all on v1 features with the BAR-ATH-calibrated noise (data/models/degrade_params.json):
  current      stage-1 LightGBM + Viterbi (what MyTeam_v1 is built from)
  +stack       + stage-2 stacked sequential learning on out-of-fold probabilities
  tau0.5       trained on labels where touches < 0.5 s between two spells of the other team are
               merged (CHALLENGE.md: a touch without control is not a change of possession)
  tau0.5+stack both

Because the true Metrica definition is unknown, every variant is scored against BOTH the raw IDSSE
labels and the tau-merged ones; the decision rule is the best worst case.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import container as C, evaluate_external as ev, features, idsse, labels, stacking  # noqa: E402

pm = C.pm

# Params
TAU = 0.5
PARAMS_PATH = os.path.join(ROOT, "data", "models", "degrade_params.json")


def pooled_spell(res, raw_by_name, definition):
    gts, prs = [], []
    sep = np.full(C.HALF_GAP, C.X, np.int8)
    for d, pred in zip(res["data"], res["preds"]):
        gt = raw_by_name[d["m"]["name"].rsplit("_deg", 1)[0]]
        if definition == "tau":
            gt = labels.merge_sandwiched(gt, TAU)
        gts += [gt, sep]
        prs += [pred, sep]
    gt, pr = np.concatenate(gts), np.concatenate(prs)
    return pm.spell_scores(gt, pr, pm.valid_runs(gt))["mean_f1"]


def main():
    t0 = time.time()
    params = json.load(open(PARAMS_PATH)) if os.path.isfile(PARAMS_PATH) else None
    quiet = lambda s: None  # noqa: E731
    raw_by_name = {m["name"]: m["label"] for m in idsse.load_all()}

    variants = {}
    variants["current"] = ev.run(feature_fn=features.build_features_v1, degrade_params=params,
                                 log=quiet)
    print("[{:4.0f}s] current".format(time.time() - t0))
    variants["+stack"] = stacking.lomo(variants["current"])
    print("[{:4.0f}s] +stack".format(time.time() - t0))
    variants["tau0.5"] = ev.run(feature_fn=features.build_features_v1,
                                label_fn=labels.harmonised(TAU), degrade_params=params, log=quiet)
    print("[{:4.0f}s] tau0.5".format(time.time() - t0))
    variants["tau0.5+stack"] = stacking.lomo(variants["tau0.5"])
    print("[{:4.0f}s] tau0.5+stack".format(time.time() - t0))

    base = [r["spellF1"] for r in variants["current"]["rows"]]
    print("\n{:<14} {:>9} {:>9} {:>7} {:>8} {:>7} {:>10} {:>9}".format(
        "variant", "vs raw", "vs tau", "worst", "macroF1", "poss", "penalties", "wins/7"))
    print("{:<14} {:>9} {:>9} {:>7} {:>8} {:>7}".format("", "spellF1", "spellF1", "case", "", "(gt 3061)"))
    for name, res in variants.items():
        a, b = pooled_spell(res, raw_by_name, "raw"), pooled_spell(res, raw_by_name, "tau")
        wins = sum(r["spellF1"] > x for r, x in zip(res["rows"], base))
        print("{:<14} {:>9.4f} {:>9.4f} {:>7.4f} {:>8.4f} {:>7d} {:>10} {:>9}".format(
            name, a, b, min(a, b), res["pooled"]["macro_f1"], res["pooled"]["n_pred_spells"],
            "{}/{}".format(*res["best_pair"]), "-" if name == "current" else wins))
    p = variants["current"]["pooled"]
    print("\ncurrent pooled interval (vs raw): {:.4f} .. {:.4f} -- gaps smaller than ~0.03 are noise".format(
        p["spell_bootstrap_low"], p["spell_bootstrap_high"]))
    print("tau-merged ground truth has {} possessions".format(
        sum(len(pm.possession_spells(labels.merge_sandwiched(g, TAU), pm.valid_runs(g)))
            for g in raw_by_name.values())))
    os.makedirs(os.path.join(ROOT, "data", "external", "idsse", "oof"), exist_ok=True)
    json.dump({k: {"best_pair": v["best_pair"], "pooled_spellF1": v["pooled"]["spell_f1_mean"]}
               for k, v in variants.items()},
              open(os.path.join(ROOT, "data", "external", "idsse", "oof", "experiments.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
