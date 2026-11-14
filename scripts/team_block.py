"""Task 14: the supervised model with the team feature block (possession/team_features.py).

    uv run python scripts/team_block.py check                  # rows = the package's rows + block
    uv run python scripts/team_block.py train [--features v1+team] [--folds-only|--final-only]
    uv run python scripts/team_block.py gain [--features v1+team] [--top 20]

`check` builds one fold's training matrix from the real prepared IDSSE matches and verifies that
its first columns and its labels are exactly the package's (`ev.stack`), and that the appended
columns are the block of the same frames. `train` trains (or loads) the final model and the 14
fold models (7 matches x noise seeds 0 and 1, each fold predicting its held-out match in both
validation input regimes), one at a time; everything is cached under .cache/team_features/.
`gain` prints the share of the final model's total gain that falls on the block. Aggregates of
the open training data only.
"""

import argparse
import time

import numpy as np

from possession import supervised
from possession import team_features as tf


def check(seed: int = 0, held_out: int = 0) -> None:
    params, feature_fn, train_cfg = supervised._setup()
    from solution import container as package_container
    from solution import evaluate_external as ev
    from solution import idsse

    prepared = ev.prepare(
        idsse.load_all(),
        feature_fn,
        supervised.label_function("raw"),
        seed=seed,
        degrade_params=params,
        log=lambda text: None,
    )
    blocks = supervised._team_blocks(prepared)
    x, y = supervised._team_stack(prepared, blocks, skip=held_out)
    x_ref, y_ref = ev.stack(prepared, skip=held_out)
    width = x_ref.shape[1]
    assert x.dtype == x_ref.dtype and x.shape == (len(x_ref), width + len(tf.feature_names()))
    assert np.array_equal(x[:, :width], x_ref, equal_nan=True), "package columns differ"
    assert np.array_equal(y, y_ref), "labels differ"
    first = 1 if held_out == 0 else 0  # the first match in the matrix
    label = prepared[first]["m"]["label"]
    rows = np.flatnonzero(label != package_container.X)[:: ev.TRAIN_STRIDE]
    block = np.asarray(blocks[first]["orig"])[rows]
    assert np.array_equal(x[: len(rows), width:], block, equal_nan=True), "block misplaced"
    swap = np.asarray(blocks[first]["swap"])[rows]
    assert np.array_equal(x[len(rows) : 2 * len(rows), width:], swap, equal_nan=True)
    weights = ev.stack_weights(prepared, held_out, ev.TRAIN_STRIDE, dict(ev.DEFAULT_CFG))
    assert len(weights) == len(y)
    print(f"check passed: {x.shape[0]} rows, {width} package columns + {x.shape[1] - width} block")


def train(features: str, folds: bool, final: bool) -> None:
    t0 = time.time()
    if final:
        supervised.model("raw", "idsse", "none", features)
        print(f"final model of {features}: {time.time() - t0:.0f} s", flush=True)
    if folds:
        for seed in (0, 1):
            t1 = time.time()
            out = supervised._team_folds("raw", seed, features)
            print(f"folds of seed {seed}: {len(out)} predictions, {time.time() - t1:.0f} s")
    print(f"total {time.time() - t0:.0f} s")


def gain(features: str, top: int) -> None:
    booster, _ = supervised.model("raw", "idsse", "none", features)
    names = tf.feature_names()
    values = np.asarray(booster.feature_importance("gain"), dtype=float)
    width = len(values) - len(names)
    share = values[width:].sum() / values.sum()
    print(f"{features}: {len(values)} features, {width} of the package's + {len(names)} block")
    print(f"share of the total gain on the block: {share:.4f}")
    print(f"block features used in at least one split: {int((values[width:] > 0).sum())}")
    ranks = np.argsort(-values)
    block_ranks = [r for r, j in enumerate(ranks) if j >= width]
    print(f"rank of the best block feature among all: {block_ranks[0] + 1 if block_ranks else '-'}")
    print(f"top {top} block features by gain (share of the total):")
    for j in [j for j in ranks if j >= width][:top]:
        print(f"  {names[j - width]:<28} {values[j] / values.sum():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "train", "gain"])
    parser.add_argument("--features", default="v1+team")
    parser.add_argument("--folds-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.action == "check":
        check()
    elif args.action == "train":
        train(args.features, folds=not args.final_only, final=not args.folds_only)
    else:
        gain(args.features, args.top)


if __name__ == "__main__":
    main()
