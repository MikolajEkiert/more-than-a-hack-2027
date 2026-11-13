"""`possession` command line. Run `uv run possession --help`.

    possession methods                     what is already implemented (check before writing code)
    possession run configs/x.toml          produce, validate, diagnose and log one submission
    possession log                         the experiment log as a table
    possession inspect {ball,players,tagging,labels} [--game G] [--csv F] [--json]
                                           aggregate statistics of the data or of a submission
                                           (rates, percentiles, histograms; never rows)
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

    p = sub.add_parser("log", help="show the experiment log")
    p.add_argument("-n", type=int, default=30)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("inspect", help="aggregate statistics of the data (never prints rows)")
    p.add_argument("topic", choices=["ball", "players", "tagging", "labels"])
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
