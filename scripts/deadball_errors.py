"""Where a configuration gets stoppages wrong (task 11, docs/FUSION.md section 10).

    uv run python scripts/deadball_errors.py [SET ...] [--config C] [--compare DIR]

SET is any of pff, idsse, metrica_label_free, metrica_match_like (default: all four). For each set:

- the pooled dead-ball error split of the configuration (`inspect_data.summarise_deadball`):
  missed and invented stoppages, whether the ball was tracked where one was invented, edge lags;
- its ball regime (`inspect_data.regime`, averaged over conditions): P(ball missing | dead or
  live) and how often a tracked ball moves during true and during invented stoppages;
- the oracle bound: true stoppages, the configuration's own team evidence and decoder.

`--compare DIR` adds the error split of saved label files `DIR/<name>/<set>/<stem>.npz` (key
`labels`) for every <name> found, e.g. predictions of other models kept outside the repository.
Aggregates only; nothing is logged.
"""

import argparse
from pathlib import Path

import numpy as np

from possession import config as config_mod
from possession import inspect_data, pipeline, stages, validation  # noqa: F401  (registers methods)
from possession.stages.decode import switch_cost

SETS = {
    "pff": {"suite": "pff"},
    "idsse": {"suite": "idsse", "seeds": (0, 1)},
    "metrica_label_free": {"suite": "metrica", "profile": "label_free"},
    "metrica_match_like": {"suite": "metrica", "profile": "match_like"},
}
SURE = 1e-3  # probability given to the wrong state by the oracle


def oracle_labels(match, truth, p_home, decode_params) -> np.ndarray:
    """Decode with the true stoppages as ball-state evidence and the given team evidence."""
    p_alive = np.where(truth == "D", SURE, 1 - SURE)
    labels = np.asarray(switch_cost(match, p_alive, p_home, **decode_params)).astype("<U1").copy()
    labels[~match.graded] = "X"
    return labels


def mean_score(labels, truths, readings) -> float:
    return float(np.mean([validation.score(labels, truths[r])["spellF1"] for r in readings]))


def run_set(name, cfg, compare_dir):
    kw = SETS[name]
    readings = validation.rule_readings(kw["suite"])
    if cfg.stages["decode"].method != "switch_cost":
        raise SystemExit("the oracle needs a switch_cost decoder")
    decode_params = cfg.stages["decode"].params
    parts, regimes, scores, oracle = [], [], [], []
    others: dict[str, list] = {}
    for game, seed, match, truths in validation.conditions(**kw):
        truth = np.asarray(truths[readings[0]]).astype("<U1")  # stoppages are the same in all
        result = pipeline.run(cfg, match)
        parts.append(inspect_data.deadball_parts(result.labels, truth, match))
        regimes.append(inspect_data.regime(match, result.labels, truth))
        scores.append(mean_score(result.labels, truths, readings))
        oracle.append(mean_score(oracle_labels(match, truth, result.p_home, decode_params),
                                 truths, readings))  # fmt: skip
        stem = f"{game}_s{seed}" if seed is not None else str(game)
        if compare_dir:
            for path in sorted(Path(compare_dir).glob(f"*/{name}/{stem}.npz")):
                labels = np.load(path)["labels"].astype("<U1")
                others.setdefault(path.parent.parent.name, []).append(
                    inspect_data.deadball_parts(labels, truth, match)
                )
    return parts, regimes, scores, oracle, others


ROW = ("true_stoppages", "missed_stoppages_share", "false_stoppages",
       "false_stoppages_ball_tracked_share", "start_on_time_share", "end_on_time_share",
       "missed_share_of_true_dead_frames", "false_share_of_true_live_frames")  # fmt: skip
REGIME = ("ball_missing_given_dead", "ball_missing_given_live", "pred_dead_ball_tracked_share",
          "true_dead_ball_tracked_share", "true_dead_tracked_moving_share",
          "false_dead_tracked_moving_share", "pred_dead_tracked_moving_share",
          "running_given_true_dead", "running_given_true_live", "running_given_true_dead_tracked",
          "running_given_true_live_tracked", "running_given_pred_dead_tracked",
          "running_given_ball_missing")  # fmt: skip


def show(name, parts, regimes, scores, oracle, others):
    print(f"\n=== {name}: {len(scores)} conditions")
    print(f"spellF1 {np.mean(scores):.4f}   oracle (true stoppages) {np.mean(oracle):.4f}"
          f"   bound {np.mean(oracle) - np.mean(scores):+.4f}")  # fmt: skip
    print("source".ljust(22) + " ".join(k[:14].rjust(14) for k in ROW) + "   start lag p50")
    for source, ps in [("config", parts)] + sorted(others.items()):
        s = inspect_data.summarise_deadball(ps)
        cells = " ".join(("" if s[k] is None else f"{s[k]:.3f}" if isinstance(s[k], float)
                          else str(s[k])).rjust(14) for k in ROW)  # fmt: skip
        print(
            source[:22].ljust(22) + cells + f"   {s['start_lag_s'].get('p50', float('nan')):+.2f} s"
        )
    s = inspect_data.summarise_deadball(parts)
    print("missed stoppages by length:", {k: v for k, v in s["missed_by_length"].items()})
    reg = {k: np.mean([r[k] for r in regimes if r.get(k) is not None] or [np.nan]) for k in REGIME}
    print("regime:", {k: round(float(v), 3) for k, v in reg.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sets", nargs="*", default=list(SETS))
    parser.add_argument("--config", default="configs/hybrid_supervised.toml")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    cfg = config_mod.load(args.config)
    for name in args.sets:
        show(name, *run_set(name, cfg, args.compare))


if __name__ == "__main__":
    main()
