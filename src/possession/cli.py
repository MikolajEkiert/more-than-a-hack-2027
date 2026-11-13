"""`possession` command line. Run `uv run possession --help`.

    possession methods                     what is already implemented (check before writing code)
    possession run configs/x.toml          produce, validate, diagnose and log one submission
    possession log                         the experiment log as a table
    possession inspect {ball,players,gaps,tagging,labels} [--game G] [--csv F] [--json]
                                           aggregate statistics of the data or of a submission
                                           (rates, percentiles, histograms; never rows)
    possession validate CONFIG... [--profile P] [--against CONFIG] [--games a,b] [--seeds 0,1,2]
                                           score configs on other, labelled matches
    possession calibrate [--write]         fit a degradation profile to the published baselines
    possession submit configs/x.toml --team NAME
                                           final file: run twice, require identical bytes,
                                           validate with the official script, write submission/

Output is aggregates only (counts, durations, scores). It never prints frame-level data, so it is
safe to show to an AI agent even when the data guard is on.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from possession import registry
from possession._paths import EVALUATION_DIR, REPO_ROOT, RESULTS_DIR, SUBMISSION_DIR


def cmd_methods(args) -> int:
    for stage in registry.STAGES:
        print(f"[{stage}]")
        for m in registry.methods(stage):
            params = ", ".join(f"{k}={v!r}" for k, v in m.params.items())
            print(f"  {m.name:<24} {m.summary}")
            print(f"  {'':<24} params: {params or '-'}   ({m.fn.__module__})")
        print()
    return 0


def cmd_inspect(args) -> int:
    import json

    from possession import inspect_data

    if args.topic == "labels":
        if not args.csv:
            raise SystemExit("inspect labels needs --csv PATH")
        result = inspect_data.labels(args.csv, args.game)
    else:
        from possession.loader import load_match

        result = getattr(inspect_data, args.topic)(load_match(args.game, verbose=False))
    print(json.dumps(result, indent=2) if args.json else inspect_data.render(result))
    return 0


def _produce(config_path: str, out_dir: Path, rebuild: bool = False):
    from possession import config as config_mod
    from possession import pipeline, scoring
    from possession.loader import load_match

    cfg = config_mod.load(config_path)
    if cfg.name.startswith("baseline_"):
        # evaluation/results/baseline_*.csv are organiser files (hash-protected); never overwrite.
        raise SystemExit(
            f"FAIL  config name {cfg.name!r}: 'baseline_*' is reserved for the organisers"
        )
    started = time.time()
    match = load_match(cfg.game, rebuild=rebuild)
    result = pipeline.run(cfg, match)
    reason = scoring.validate(result.labels, match.n_frames)
    if reason:
        raise SystemExit(f"FAIL  {cfg.name}: {reason}")
    sha = pipeline.write_csv(out_dir / f"{cfg.name}.csv", result.labels)
    diag = scoring.diagnostics(result.labels, match.half, match.fps)
    sanity = scoring.public_sanity(result.labels, match.game)
    return cfg, sha, diag, sanity, time.time() - started


def cmd_run(args) -> int:
    from possession import experiments

    out_dir = Path(args.out)
    cfg, sha, diag, sanity, seconds = _produce(args.config, out_dir, args.rebuild_cache)
    print(f"OK    {cfg.name}  ->  {_rel(out_dir / (cfg.name + '.csv'))}")
    print(f"      sha256 {sha[:16]}   {seconds:.1f} s")
    print(
        f"      {diag['n_possessions']} possessions, mean {diag['mean_possession_s']} s, "
        f"home share {diag['home_share']}, graded label share {diag['label_share_graded']}"
    )
    if sanity:
        print(
            f"      public 5-min sanity: spellF1 {sanity['spellF1']}  macroF1 {sanity['macroF1']}"
            f"  ({sanity['n_pred_possessions']} vs {sanity['n_gt_possessions']} possessions)"
            "  <- sanity only"
        )
    if not args.no_log:
        path = experiments.record(cfg, sha, diag, sanity, seconds, note=args.note or "")
        print(f"      logged {_rel(path)}")
    return 0


def _csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _evaluate(config_path: str, games, seeds, profile):
    """(config, cells, seconds, last run's diagnostics) of one config on the external matches."""
    from possession import config as config_mod
    from possession import pipeline, scoring, validation

    cfg = config_mod.load(config_path)
    last = {}

    def run(match):
        labels = pipeline.run(cfg, match).labels
        last["diagnostics"] = scoring.diagnostics(labels, match.half, match.fps)
        return labels

    started = time.time()
    cells = validation.evaluate(run, games=games, seeds=seeds, profile=profile)
    return cfg, cells, time.time() - started, last.get("diagnostics", {})


def cmd_validate(args) -> int:
    """Score configs on other, labelled matches (never the competition match)."""
    from possession import experiments, validation

    games = _csv(args.games) if args.games else list(validation.GAMES)
    seeds = [int(seed) for seed in _csv(args.seeds)]
    profile = None if args.clean else args.profile
    condition = "clean" if args.clean else f"degraded(profile={profile}, seeds={seeds})"
    print(f"external matches {games}, {condition}; spellF1 under each label convention")
    header = f"{'config':<28} {'mean':>6} " + " ".join(f"{c:>10}" for c in validation.CONVENTIONS)
    print(header + "   " + "  ".join(f"{g[-3:]}" for g in games) + "   poss pred/gt (last)")

    results = {}
    reference = _evaluate(args.against, games, seeds, profile) if args.against else None
    for path in args.configs:
        cfg, cells, seconds, diag = _evaluate(path, games, seeds, profile)
        results[cfg.name] = cells
        summary = validation.summarise(cells)
        last = cells[-1]
        print(
            f"{cfg.name:<28} {summary['mean']:>6.4f} "
            + " ".join(f"{summary['by_convention'][c]:>10.4f}" for c in validation.CONVENTIONS)
            + "   "
            + "  ".join(f"{summary['by_game'][g]:.3f}"[1:] for g in games)
            + f"   {last['n_pred_possessions']}/{last['n_gt_possessions']}   {seconds:.0f} s"
        )
        external = {
            "games": games,
            "seeds": seeds,
            "profile": profile,
            "condition": condition,
            "spellF1": summary["mean"],
            "summary": summary,
            "macroF1": validation.summarise(cells, "macroF1")["mean"],
            "cells": cells,
        }
        if reference:
            verdict = validation.compare(cells, reference[1])
            external["against"] = {"name": reference[0].name, **verdict}
            print(
                f"    vs {reference[0].name}: gain {verdict['gain']:+.4f}, by convention "
                f"{verdict['gain_by_convention']}, wins {verdict['games_won']}/{verdict['n_games']}"
                f" matches -> {'ACCEPT' if verdict['accepted'] else 'reject'}"
            )
        if not args.no_log:
            experiments.record(
                cfg, "", diag, None, seconds, note=args.note or "", external=external
            )
    return 0


def cmd_calibrate(args) -> int:
    """Fit a degradation profile to the organisers' published full-match baseline scores."""
    from possession import validation

    games = _csv(args.games) if args.games else list(validation.GAMES)
    seeds = [int(seed) for seed in _csv(args.seeds)]
    base = validation.baseline_residual(games, seeds, profile="label_free")
    print(f"published (spellF1, macroF1) by window: {validation.PUBLISHED_FULL_MATCH}")
    print(f"label_free profile: rmse {base['rmse']}  touch {base['touch']['rmse']}"
          f"  controlled {base['controlled']['rmse']}")  # fmt: skip

    def progress(row):
        print(f"  rmse {row['rmse']:.4f}  {row['overrides']}", flush=True)

    rows = validation.calibrate(games, seeds, progress=progress)
    best = rows[0]
    print(f"best: rmse {best['rmse']}  touch {best['touch']['rmse']}"
          f"  controlled {best['controlled']['rmse']}  {best['overrides']}")  # fmt: skip
    if args.write:
        params = {**validation.degradation_profile("label_free"), **best["overrides"]}
        evidence = {
            "games": games,
            "seeds": seeds,
            "published_full_match": {str(k): v for k, v in validation.PUBLISHED_FULL_MATCH.items()},
            "label_free_profile": base,
            "fitted": {k: v for k, v in best.items() if k != "overrides"},
            "grid": {k: list(v) for k, v in validation.CALIBRATION_GRID.items()},
            "runners_up": [{"rmse": r["rmse"], "overrides": r["overrides"]} for r in rows[1:4]],
        }
        source = (
            "Fitted by `possession calibrate` so that the four organiser baselines score on the"
            " degraded external matches what EVALUATION.md publishes for them on the full match"
            " (mean over both label conventions of the RMSE to the eight published numbers)."
            " Uses published documentation, never the ground truth files."
        )
        path = validation.write_profile("published_baselines", params, source, evidence)
        print(f"wrote {_rel(path)}")
    return 0


def cmd_log(args) -> int:
    from possession import experiments

    entries = experiments.load_all()
    if args.n:
        entries = entries[-args.n :]
    print(experiments.table(entries))
    return 0


def cmd_submit(args) -> int:
    cfg, sha, diag, _, _ = _produce(args.config, RESULTS_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        print("re-running from a fresh parse to prove the file is reproducible ...", flush=True)
        _, sha_again, _, _, _ = _produce(args.config, Path(tmp), rebuild=True)
    if sha != sha_again:
        print(f"FAIL  not reproducible: {sha[:16]} != {sha_again[:16]}")
        return 1

    SUBMISSION_DIR.mkdir(exist_ok=True)
    target = SUBMISSION_DIR / f"{args.team}.csv"
    shutil.copyfile(RESULTS_DIR / f"{cfg.name}.csv", target)
    sys.stdout.flush()
    check = subprocess.run(
        [sys.executable, str(EVALUATION_DIR / "validate_submission.py"), str(target)]
    )
    if check.returncode:
        return check.returncode
    print(f"\nSUBMISSION  {_rel(target)}  sha256 {sha}")
    print(f"config {args.config}  commit {_commit()}  -> record both in docs/TECHNICAL_REPORT.md")
    return 0


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _commit() -> str:
    from possession import experiments

    return experiments.git("rev-parse", "--short", "HEAD")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="possession", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("methods", help="list registered methods").set_defaults(fn=cmd_methods)

    p = sub.add_parser("run", help="produce, validate and log one submission")
    p.add_argument("config")
    p.add_argument("--out", default=str(RESULTS_DIR))
    p.add_argument("--note", help="one line for the experiment log: what this run tests")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("validate", help="score configs on other, labelled matches")
    p.add_argument("configs", nargs="+")
    p.add_argument("--games", default="", help="comma-separated; default: all external matches")
    p.add_argument("--seeds", default="0,1,2", help="seeds of the degradation")
    p.add_argument("--profile", default="label_free", help="a name in configs/degradation/")
    p.add_argument("--clean", action="store_true", help="do not degrade the tracking")
    p.add_argument("--against", help="config to compare with under the acceptance rule")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("calibrate", help="fit a degradation profile to the published baselines")
    p.add_argument("--games", default="")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--write", action="store_true", help="write configs/degradation/")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("log", help="show the experiment log")
    p.add_argument("-n", type=int, default=30)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("inspect", help="aggregate statistics of the data (never prints rows)")
    p.add_argument("topic", choices=["ball", "players", "gaps", "tagging", "labels"])
    p.add_argument("--game", default=None, help="default: games.GAME")
    p.add_argument("--csv", help="submission CSV (topic labels)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("submit", help="final reproducible submission file")
    p.add_argument("config")
    p.add_argument("--team", required=True, help="team name; the file becomes <team>.csv")
    p.set_defaults(fn=cmd_submit)

    args = parser.parse_args(argv)
    if getattr(args, "game", 0) is None:
        import games

        args.game = games.GAME
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
