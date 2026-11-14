"""Competitor-review adjustments, measured on IDSSE (leave-one-match-out). Public window unused.

    python -m solution.experiments_v2

Stage 1 (training):  base | +weights | factorized | factorized+weights
Decoder (on the best stage 1): minimum dead-ball length x touch snapping, then + boundary model
Labels: 2 s "controlled possession" relabel, scored against both definitions (worst case).

A change is adopted only if the pooled gain is >= ADOPT and it wins on >= 5 of 7 held-out matches.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import (boundary, container as C, decode, evaluate_external as ev, features,  # noqa: E402
                      idsse, labels)

pm = C.pm

# Params
FEATURE_FN = features.build_features_v1
STAGE1 = {"base": {}, "+weights": {"weights": True}, "factorized": {"factorized": True},
          "factorized+weights": {"factorized": True, "weights": True}}
MIN_DEAD = (0, 12, 25, 50)        # frames
SNAP = (0, 3, 6)                  # frames
CONTROL_S = 2.0
ADOPT = 0.005
OUT = os.path.join(ROOT, "data", "external", "idsse", "oof", "experiments_v2.json")


def per_match(res):
    return np.array([r["spellF1"] for r in res["rows"]])


def line(name, res, ref=None):
    p = res["pooled"]
    s = per_match(res)
    extra = ""
    if ref is not None:
        d = s - per_match(ref)
        extra = "  {:+.4f} pooled  {:+.4f} per match  wins {}/{}".format(
            p["spell_f1_mean"] - ref["pooled"]["spell_f1_mean"], d.mean(), int(np.sum(d > 0)), len(d))
    print("{:<34} {:.4f}  [{:.4f} .. {:.4f}]  macroF1 {:.4f}  poss {:>5}{}".format(
        name, p["spell_f1_mean"], p.get("spell_bootstrap_low") or np.nan,
        p.get("spell_bootstrap_high") or np.nan, p["macro_f1"], p["n_pred_spells"], extra))


def adopt(res, ref):
    d = per_match(res) - per_match(ref)
    return (res["pooled"]["spell_f1_mean"] - ref["pooled"]["spell_f1_mean"] >= ADOPT
            and np.sum(d > 0) >= 5)


def main():
    t0 = time.time()
    path = os.path.join(ROOT, "data", "models", "degrade_params.json")
    params = json.load(open(path)) if os.path.isfile(path) else None
    q = lambda s: None  # noqa: E731
    out = {}

    print("== stage 1 (training variants), default decoder")
    s1 = {}
    for name, cfg in STAGE1.items():
        s1[name] = ev.run(feature_fn=FEATURE_FN, degrade_params=params, train_cfg=cfg,
                          bootstrap=200, log=q)
        line(name, s1[name], None if name == "base" else s1["base"])
    best1 = "base"
    for name in STAGE1:
        if name != "base" and adopt(s1[name], s1["base"]) and \
                s1[name]["pooled"]["spell_f1_mean"] > s1[best1]["pooled"]["spell_f1_mean"]:
            best1 = name
    print("-> stage 1:", best1, "[{:.0f}s]".format(time.time() - t0))
    out["stage1"] = best1

    # features (memory-mapped) for touch snapping and the boundary model
    data = ev.prepare(idsse.load_all(), FEATURE_FN, labels.raw, degrade_params=params, log=q)
    names = data[0]["names"]
    th, ta = names.index("touch_h"), names.index("touch_a")
    touches = [(np.asarray(d["X"][:, th], np.float32), np.asarray(d["X"][:, ta], np.float32)) for d in data]
    pb = boundary.oof(data, FEATURE_FN, log=q)
    for d in data:
        d.pop("X")
        d.pop("Xs", None)
    probs = s1[best1]["probs"]
    halves = [d["m"]["half"] for d in data]

    print("\n== decoder rules on '{}' (min dead-ball frames x snap radius)".format(best1))
    plain = [decode.decode(probs[k], halves[k], ev.PAIRS) for k in range(len(data))]
    ref = s1[best1]
    best_post, best_res = (0, 0), ref
    for md in MIN_DEAD:
        for sr in SNAP:
            if md == 0 and sr == 0:
                continue
            def post(k, prob, half, md=md, sr=sr):
                return np.stack([decode.snap(decode.min_dead(p, half, md), *touches[k], sr)
                                 for p in plain[k]])
            r = ev.score(probs, data, ev.PAIRS, bootstrap=0, decode_fn=post)
            line("min_dead {:>2} snap {}".format(md, sr), r, ref)
            if adopt(r, ref) and r["pooled"]["spell_f1_mean"] > best_res["pooled"]["spell_f1_mean"]:
                best_post, best_res = (md, sr), r
    print("-> decoder rules:", best_post)
    out["min_dead"], out["snap"] = best_post

    print("\n== + boundary-aware decoder (P(hand-over) -> switch cost)")
    md, sr = best_post

    def bdec(k, prob, half):
        paths = boundary.decode_with(prob, half, pb[k])
        return np.stack([decode.snap(decode.min_dead(p, half, md), *touches[k], sr) for p in paths])
    rb = ev.score(probs, data, boundary.CONFIGS, bootstrap=200, decode_fn=bdec)
    line("boundary decoder", rb, best_res)
    use_b = adopt(rb, best_res)
    out["boundary"] = bool(use_b)
    out["boundary_config"] = rb["best_pair"]
    final = rb if use_b else best_res
    print("-> boundary decoder:", use_b, "| best config", rb["best_pair"])

    print("\n== labels: {} s controlled possession (robustness, both definitions)".format(CONTROL_S))
    rc = ev.run(feature_fn=FEATURE_FN, label_fn=labels.controlled(CONTROL_S), degrade_params=params,
                train_cfg=STAGE1[best1], bootstrap=0, log=q)
    raw_by = {m["name"]: m["label"] for m in idsse.load_all()}

    def pooled_vs(res, definition):
        gts, prs, sep = [], [], np.full(C.HALF_GAP, C.X, np.int8)
        for d, pred in zip(res["data"], res["preds"]):
            gt = raw_by[d["m"]["name"].rsplit("_deg", 1)[0]]
            gt = labels.controlled_possession(gt, CONTROL_S) if definition == "ctl" else gt
            gts += [gt, sep]
            prs += [pred, sep]
        gt, pr = np.concatenate(gts), np.concatenate(prs)
        return pm.spell_scores(gt, pr, pm.valid_runs(gt))["mean_f1"]
    a_raw, a_ctl = pooled_vs(s1[best1], "raw"), pooled_vs(s1[best1], "ctl")
    b_raw, b_ctl = pooled_vs(rc, "raw"), pooled_vs(rc, "ctl")
    print("trained on raw:        vs raw {:.4f}  vs 2s {:.4f}  worst {:.4f}".format(a_raw, a_ctl, min(a_raw, a_ctl)))
    print("trained on 2s control: vs raw {:.4f}  vs 2s {:.4f}  worst {:.4f}".format(b_raw, b_ctl, min(b_raw, b_ctl)))
    out["controlled"] = bool(min(b_raw, b_ctl) > min(a_raw, a_ctl) + ADOPT)

    print("\nFINAL on IDSSE: spellF1 {:.4f} (base {:.4f}) | choices {}  [{:.0f}s]".format(
        final["pooled"]["spell_f1_mean"], s1["base"]["pooled"]["spell_f1_mean"], out, time.time() - t0))
    out["final_pooled"] = final["pooled"]["spell_f1_mean"]
    out["final_pair"] = final["best_pair"]
    json.dump(out, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
