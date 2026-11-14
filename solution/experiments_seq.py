"""v1 vs v2 (v1 + sequence features), each with the configuration experiments_v2 chose. IDSSE only.

    python -m solution.experiments_seq
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import boundary, decode, evaluate_external as ev, features, features_seq, idsse, labels  # noqa: E402
from solution.experiments_v2 import ADOPT, OUT as V2_OUT, STAGE1, line, per_match  # noqa: E402

OUT = os.path.join(ROOT, "data", "external", "idsse", "oof", "experiments_seq.json")


def evaluate(feature_fn, params, choice, q):
    res = ev.run(feature_fn=feature_fn, degrade_params=params, train_cfg=STAGE1[choice["stage1"]],
                 bootstrap=0, log=q)
    data = ev.prepare(idsse.load_all(), feature_fn, labels.raw, degrade_params=params, log=q)
    names = data[0]["names"]
    th, ta = names.index("touch_h"), names.index("touch_a")
    touches = [(np.asarray(d["X"][:, th], np.float32), np.asarray(d["X"][:, ta], np.float32)) for d in data]
    pb = boundary.oof(data, feature_fn, log=q) if choice["boundary"] else None
    for d in data:
        d.pop("X")
        d.pop("Xs", None)
    md, sr = choice["min_dead"], choice["snap"]
    if pb is not None:
        def dec(k, prob, half):
            return np.stack([decode.snap(decode.min_dead(p, half, md), *touches[k], sr)
                             for p in boundary.decode_with(prob, half, pb[k])])
        pairs = boundary.CONFIGS
    else:
        def dec(k, prob, half):
            return np.stack([decode.snap(decode.min_dead(p, half, md), *touches[k], sr)
                             for p in decode.decode(prob, half, ev.PAIRS)])
        pairs = ev.PAIRS
    out = ev.score(res["probs"], data, pairs, bootstrap=200, decode_fn=dec)
    out["gain"], out["names"] = res["gain"], res["names"]
    return out


def main():
    t0 = time.time()
    params = json.load(open(os.path.join(ROOT, "data", "models", "degrade_params.json")))
    choice = json.load(open(V2_OUT))
    print("configuration from experiments_v2:", {k: choice[k] for k in ("stage1", "min_dead", "snap", "boundary")})
    q = lambda s: None  # noqa: E731
    r1 = evaluate(features.build_features_v1, params, choice, q)
    line("v1", r1)
    r2 = evaluate(features_seq.build_features_v2, params, choice, q)
    line("v2 (+ sequence features)", r2, r1)
    d = per_match(r2) - per_match(r1)
    use = (r2["pooled"]["spell_f1_mean"] - r1["pooled"]["spell_f1_mean"] >= ADOPT and np.sum(d > 0) >= 5)
    top = np.argsort(-r2["gain"])[:15]
    print("v2 top features:", ", ".join(r2["names"][i] for i in top))
    print("-> features:", "v2" if use else "v1", "[{:.0f}s]".format(time.time() - t0))
    json.dump({"features": "v2" if use else "v1", "pooled_v1": r1["pooled"]["spell_f1_mean"],
               "pooled_v2": r2["pooled"]["spell_f1_mean"], "pair": r2["best_pair"] if use else r1["best_pair"]},
              open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
