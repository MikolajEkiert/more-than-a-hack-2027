"""Per-frame probabilities of the supervised branch's model, for the fusion experiments.

This script does not belong to our pipeline. It runs INSIDE a checkout of the branch
`supervised_approach` (package `solution/`), with that branch's own environment, and writes one
.npz per match that `fusion.from_file` can read (docs/FUSION.md):

    # in this repository: the tracking of every validation condition, no labels
    uv run possession export --profile label_free --out /tmp/conditions
    uv run possession export --profile published_baselines --out /tmp/conditions

    # in a worktree of the supervised branch, with requirements-solution.txt (+ scipy) installed
    git worktree add --detach ../mtah-supervised-eval origin/supervised_approach
    cd ../mtah-supervised-eval
    MTAH_DATA_DIR=<this repo>/data python -m solution.idsse            # downloads IDSSE (2.7 GB)
    MTAH_DATA_DIR=<this repo>/data python <this repo>/scripts/supervised_probs.py \\
        /tmp/conditions <this repo>/.cache/supervised

It trains that branch's final configuration exactly as its notebook does (degradation calibrated
on the unlabelled tracking of the match, v1 features, weighted LightGBM, boundary model) and
saves, for the competition match and for every exported condition:

    p_h, p_a, p_d   class probabilities per frame
    p_boundary      the boundary model's probability of a hand-over
    codes           the branch's own decoded labels (0 H, 1 A, 2 D, 3 X), without its kick-off rule

Nothing is trained on the Metrica matches, so all of them are held out for this model.
"""

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.getcwd()
for path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from solution import atd, boundary, cache, degrade, features, labels, submission  # noqa: E402
from solution import container as C  # noqa: E402


def main(exports: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    def log(text):
        print(f"[{time.time() - started:5.0f}s] {text}", flush=True)

    bar = atd.load()
    if not atd.checks(bar)["ok"]:
        raise SystemExit("the match does not load as DATA.md describes it")
    params = degrade.calibrated_params(bar)  # unlabelled statistics of the tracking only
    os.makedirs("data/models", exist_ok=True)
    with open(submission.FINAL["degrade_json"], "w") as handle:
        json.dump(params, handle)
    log(f"calibrated degradation: { {k: round(float(v), 4) for k, v in params.items()} }")

    feature_fn = features.FEATURE_SETS[submission.FINAL["features"]]
    train_cfg = {"weights": True} if submission.FINAL["weights"] else None
    model = submission.train_final(feature_fn, labels.raw, params, log=log, train_cfg=train_cfg)
    boundary_model = boundary.train_final(feature_fn, labels.raw, params)
    config = tuple(submission.FINAL["boundary"])
    log("models trained")

    def predict(match):
        key = cache.digest("match", cache.fingerprint(match))
        x, _ = features.cached(feature_fn, match, "orig", key)
        prob = model.predict(x)
        p_boundary = boundary_model.predict(x)
        codes = boundary.decode_with(prob, match["half"], p_boundary, [config])[0]
        return prob, p_boundary, codes

    def save(name, prob, p_boundary, codes):
        np.savez_compressed(
            os.path.join(out_dir, name + ".npz"),
            p_h=prob[:, 0],
            p_a=prob[:, 1],
            p_d=prob[:, 2],
            p_boundary=p_boundary,
            codes=codes,
        )
        log(f"wrote {name}")

    save(bar["name"], *predict(bar))
    for path in sorted(glob.glob(os.path.join(exports, "*.npz"))):
        with np.load(path) as data:
            n = len(data["half"])
            match = {
                "px": data["players"][:, :, 0].astype(np.float32),
                "py": data["players"][:, :, 1].astype(np.float32),
                "team": data["team_of_track"].astype(np.int8),
                "bx": data["ball"][:, 0].astype(np.float32),
                "by": data["ball"][:, 1].astype(np.float32),
                "half": data["half"].astype(np.int8),
                "label": np.full(n, C.X, np.int8),
                "pitch": data["pitch"].astype(np.float32),
                "name": os.path.basename(path)[:-4],
            }
        save(match["name"], *predict(match))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
