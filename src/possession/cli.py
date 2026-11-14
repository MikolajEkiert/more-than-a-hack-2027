"""`possession` command line. Run `uv run possession --help`.

    possession methods                     what is already implemented (check before writing code)
    possession run configs/x.toml          produce, validate, diagnose and log one submission
    possession log                         the experiment log as a table
    possession inspect {ball,players,gaps,tagging,kickoffs,labels,regime} [--game G] [--csv F]
                       [--config C] [--json]
                                           aggregate statistics of the data or of a submission
                                           (rates, percentiles, histograms; never rows)
    possession validate CONFIG... [--suite S] [--profile P] [--against CONFIG] [--games a,b]
                                           score configs on other, labelled matches
    possession decide V1.toml V3.toml      choose between two versions by the rule fixed in
                                           docs/FUSION.md section 8 (evidence, vetoes, footprint)
    possession calibrate [--target match] [--write]
                                           fit a degradation profile (to label-free statistics
                                           of the match, or to the published baseline scores)
    possession export [--profile P]        write the validation conditions' tracking to .npz
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
    elif args.topic == "regime":
        from possession import config as config_mod
        from possession import pipeline
        from possession.loader import load_match

        match = load_match(args.game, verbose=False)
        labels = pipeline.run(config_mod.load(args.config), match).labels
        result = inspect_data.regime(match, labels)
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


def _evaluate(config_path: str, games, seeds, profile, suite="metrica"):
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
    cells = validation.evaluate(run, games=games, seeds=seeds, profile=profile, suite=suite)
    return cfg, cells, time.time() - started, last.get("diagnostics", {})


def _package_decoder(games, seeds):
    """Cells of the supervised package's own decoder on the IDSSE matches, for comparison."""
    from possession import supervised, validation

    def run(match):
        return match._cache[supervised.PACKAGE_LABELS]("raw")

    return validation.evaluate(run, games=games, seeds=seeds, suite="idsse")


def _row(name, cells, readings, seconds=None):
    """One line of the validation table: mean, each reading, each match, possessions, stoppages."""
    from possession import validation

    summary = validation.summarise(cells, readings=readings)
    counted = [c for c in cells if c["convention"] in summary["readings"]]
    ratio = sum(c["n_pred_possessions"] for c in counted) / max(
        sum(c["n_gt_possessions"] for c in counted), 1
    )
    seen_live = validation.summarise(cells, "seen_stoppage_called_live", readings)["mean"]
    return (
        f"{name:<28} {summary['mean']:>6.4f} "
        + " ".join(f"{value:>13.4f}" for value in summary["by_convention"].values())
        + "   "
        + " ".join(f"{value:.3f}"[1:] for value in summary["by_game"].values())
        + f"   {ratio:>5.2f}"
        + (f"   {seen_live:>5.2f}" if seen_live is not None else "       -")
        + (f"   {seconds:.0f} s" if seconds is not None else "")
    )


def cmd_validate(args) -> int:
    """Score configs on other, labelled matches (never the competition match)."""
    from possession import experiments, validation

    suite = args.suite
    games = _csv(args.games) if args.games else None
    seeds = [int(seed) for seed in _csv(args.seeds)]
    profile = None if args.clean or suite != "metrica" else args.profile
    readings = validation.rule_readings(suite)
    if suite == "metrica":
        condition = "clean" if args.clean else f"degraded(profile={profile}, seeds={seeds})"
    else:
        condition = f"{suite}, each match scored by a model that never saw it"
    print(f"suite {suite}: {condition}; spellF1, mean over the readings {list(readings)}")

    results = []
    reference = _evaluate(args.against, games, seeds, profile, suite) if args.against else None
    for path in args.configs:
        results.append(_evaluate(path, games, seeds, profile, suite))

    first = results[0][1]
    present = list(dict.fromkeys(c["convention"] for c in first))
    names = sorted({c["game"] for c in first})
    print(
        f"{'config':<28} {'mean':>6} "
        + " ".join(f"{c:>13}" for c in present)
        + "   "
        + " ".join(f"{g[-3:]:>4}" for g in names)
        + "   pred/true possessions   stoppage with ball in view called play"
    )
    if suite == "idsse":
        print(_row("(package's own decoder)", _package_decoder(games, seeds), readings))
    for cfg, cells, seconds, diag in results:
        print(_row(cfg.name, cells, readings, seconds))
        summary = validation.summarise(cells, readings=readings)
        external = {
            "suite": suite,
            "games": names,
            "seeds": seeds,
            "profile": profile,
            "condition": condition,
            "spellF1": summary["mean"],
            "summary": summary,
            "macroF1": validation.summarise(cells, "macroF1", readings)["mean"],
            "cells": cells,
        }
        if reference:
            verdict = validation.compare(cells, reference[1], readings=readings)
            external["against"] = {"name": reference[0].name, **verdict}
            print(
                f"    vs {reference[0].name}: gain {verdict['gain']:+.4f}, by reading "
                f"{verdict['gain_by_convention']}, wins {verdict['games_won']}/{verdict['n_games']}"
                f" matches -> {'ACCEPT' if verdict['accepted'] else 'reject'}"
                + (
                    f"  (reported only: {verdict['gain_reported_only']})"
                    if verdict["gain_reported_only"]
                    else ""
                )
            )
        if not args.no_log:
            experiments.record(
                cfg, "", diag, None, seconds, note=args.note or "", external=external
            )
    return 0


def cmd_decide(args) -> int:
    """Choose between two versions by the rule of docs/FUSION.md section 8."""
    from possession import config as config_mod
    from possession import experiments, loader, pipeline, scoring, validation

    started = time.time()
    paths = {"first": args.first, "second": args.second}
    configs = {name: config_mod.load(path) for name, path in paths.items()}
    seeds = [int(seed) for seed in _csv(args.seeds)]
    weights = None
    if args.weights:
        weights = {k: float(v) for k, v in (item.split("=") for item in _csv(args.weights))}
    print(f"version 1: {configs['first'].name} ({paths['first']})")
    print(f"version 3: {configs['second'].name} ({paths['second']})")

    evidence = {}
    for profile in _csv(args.profiles):
        evidence[profile] = tuple(
            _evaluate(paths[name], None, seeds, profile, "metrica")[1] for name in paths
        )
    vetoes, unavailable = {}, []
    for suite in validation.VETO_WINS:
        try:
            vetoes[suite] = tuple(
                _evaluate(paths[name], None, seeds, None, suite)[1] for name in paths
            )
        except FileNotFoundError as error:
            unavailable.append(f"{suite}: {error}")

    match = loader.load_match(verbose=False)
    results = {name: pipeline.run(cfg, match) for name, cfg in configs.items()}
    footprint = {n: scoring.footprint(r.labels, match.half, match.fps) for n, r in results.items()}
    between = validation.agreement(match, results["first"], results["second"])
    graded = match.graded
    changed = float((results["first"].labels[graded] != results["second"].labels[graded]).mean())
    sanity = {n: scoring.public_sanity(r.labels, match.game) for n, r in results.items()}

    verdict = validation.choose(evidence, vetoes, footprint, weights)
    if unavailable:
        verdict["choose"] = "first"
        verdict["conditions"].update(
            {f"veto set available: {u.split(':')[0]}": False for u in unavailable}
        )

    print("\nevidence, Metrica matches: spellF1 of version 3 minus version 1")
    for scenario, value in verdict["scenarios"].items():
        print(f"  {scenario:<34} {value:+.4f}")
    interval = verdict["weighted_gain_interval"]
    print(
        f"  weighted mean {verdict['weighted_gain']:+.4f}"
        + (
            f"   5-95% over (match, window) blocks [{interval[0]:+.4f}, {interval[1]:+.4f}]"
            if interval
            else ""
        )
        + f"   windows won {verdict['windows_won_share']}   weights {verdict['weights']}"
    )
    print("veto sets:")
    for suite, v in verdict["vetoes"].items():
        print(
            f"  {suite:<6} gain {v['gain']:+.4f}  by reading {v['by_reading']}"
            f"  matches won {v['matches_won']}/{v['n_matches']}"
            f"  -> {'passes' if v['passes'] else 'VETO'}"
        )
    for line in unavailable:
        print(f"  not available: {line}")
    print("competition match, no labels:            version 1   version 3")
    for key in ("possessions", "possessions_per_live_minute", "mean_possession_s", "dead_share"):
        print(f"  {key:<38} {footprint['first'][key]:>9}   {footprint['second'][key]:>9}")
    blocks = {n: footprint[n]["per_live_minute_by_block"] for n in footprint}
    print(f"  per live minute by 10-minute block       {blocks['first']}   {blocks['second']}")
    print(
        f"  frames that differ {changed:.2%}; possessions of one scored against the other"
        f" {between['possession_agreement']}; same team where both say play"
        f" {between['team_agreement']}"
    )
    if all(sanity.values()):
        print(
            "  public window, sanity only (not part of the rule): "
            + ", ".join(
                f"{label} {sanity[n]['spellF1']} ({sanity[n]['n_pred_possessions']} vs"
                f" {sanity[n]['n_gt_possessions']} possessions)"
                for n, label in (("first", "version 1"), ("second", "version 3"))
            )
        )
    print("conditions for version 3:")
    for name, holds in verdict["conditions"].items():
        print(f"  [{'x' if holds else ' '}] {name}")
    chosen = "version 3" if verdict["choose"] == "second" else "version 1"
    print(f"verdict: {chosen} ({configs[verdict['choose']].name})")

    if not args.no_log:
        external = {
            "decision": {
                **verdict,
                "version_1": paths["first"],
                "version_3": paths["second"],
                "profiles": list(evidence),
                "seeds": seeds,
                "footprint": footprint,
                "between_versions": {**between, "frames_that_differ": round(changed, 4)},
                "public_window_sanity_only": sanity,
                "unavailable": unavailable,
            }
        }
        diag = scoring.diagnostics(results["second"].labels, match.half, match.fps)
        path = experiments.record(
            configs["second"],
            "",
            diag,
            None,
            time.time() - started,
            note=args.note or "",
            external=external,
        )
        print(f"logged {_rel(path)}")
    return 0


def cmd_export(args) -> int:
    """Write every validation condition's tracking to .npz, for a model that lives elsewhere."""
    import numpy as np

    from possession import validation

    games = _csv(args.games) if args.games else list(validation.GAMES)
    seeds = [int(seed) for seed in _csv(args.seeds)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = None if args.clean else args.profile
    for _game, _seed, match, _truths in validation.conditions(games, seeds, profile):
        path = out_dir / f"{match.game}.npz"
        np.savez_compressed(
            path,
            ball=match.ball,
            players=match.players,
            team_of_track=match.team_of_track,
            half=match.half,
            fps=np.array(match.fps),
            pitch=np.array(match.pitch),
        )
        print(f"wrote {_rel(path)}  ({match.n_frames} frames)")
    print("no labels are written; return per-frame probabilities named '<same name>.npz'")
    return 0


def _calibrate_to_match(args) -> int:
    """Fit the profile `match_like` to label-free statistics of the competition match."""
    from possession import config as config_mod
    from possession import loader, pipeline, validation

    config_a, config_b = args.configs.split(",")
    games = _csv(args.games) if args.games else list(validation.GAMES)
    seeds = [int(seed) for seed in _csv(args.seeds)]
    match = loader.load_match(verbose=False)
    target = validation.agreement(
        match,
        pipeline.run(config_mod.load(config_a), match),
        pipeline.run(config_mod.load(config_b), match),
    )
    print(f"competition match, no labels ({config_a} against {config_b}): {target}")

    def progress(row):
        stats = {k: row[k] for k in (*validation.AGREEMENT_KEYS, "ball_missing")}
        print(f"  loss {row['loss']:.4f}  {stats}  {row['overrides']}", flush=True)

    rows = validation.calibrate_to_match(
        config_a, config_b, target, games, seeds, processes=args.processes, progress=progress
    )
    best = rows[0]
    print(f"best: loss {best['loss']}  {best['overrides']}")
    for name in ("label_free", "published_baselines"):
        params = validation.degradation_profile(name)
        near = [r for r in rows if all(params[k] == v for k, v in r["overrides"].items())]
        if near:
            print(f"for comparison, the grid point of {name}: loss {near[0]['loss']}")
    if args.write:
        params = {**validation.degradation_profile("label_free"), **best["overrides"]}
        evidence = {
            "games": games,
            "seeds": seeds,
            "configs": [config_a, config_b],
            "target_competition_match": target,
            "fitted": {k: v for k, v in best.items() if k != "overrides"},
            "grid": {k: list(v) for k, v in validation.MATCH_GRID.items()},
            "runners_up": [{"loss": r["loss"], "overrides": r["overrides"]} for r in rows[1:4]],
        }
        source = (
            "Fitted by `possession calibrate --target match` so that three label-free statistics"
            " of our two pipelines on the degraded external matches equal the same statistics on"
            " the competition match: how often they agree on possessions, how often on the team,"
            " and how sure the team evidence is. Uses no label of the match and no published"
            " score."
        )
        path = validation.write_profile("match_like", params, source, evidence)
        print(f"wrote {_rel(path)}")
    return 0


def cmd_calibrate(args) -> int:
    """Fit a degradation profile: to label-free statistics of the match, or to the organisers'
    published full-match baseline scores."""
    from possession import validation

    if args.target == "match":
        return _calibrate_to_match(args)
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
    p.add_argument("--suite", default="metrica", choices=["metrica", "idsse", "pff"])
    p.add_argument("--clean", action="store_true", help="do not degrade the tracking")
    p.add_argument("--against", help="config to compare with under the acceptance rule")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("decide", help="choose between two versions (docs/FUSION.md section 8)")
    p.add_argument("first", help="version 1: the cautious configuration")
    p.add_argument("second", help="version 3: the bet")
    p.add_argument("--profiles", default="label_free,match_like", help="evidence profiles")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--weights", default="", help="e.g. touch=0.5,controlled=0.5 (the default)")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("export", help="write the validation conditions' tracking to .npz")
    p.add_argument("--out", default=str(REPO_ROOT / ".cache" / "exports"))
    p.add_argument("--games", default="")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--profile", default="label_free")
    p.add_argument("--clean", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("calibrate", help="fit a degradation profile")
    p.add_argument("--target", default="published", choices=["published", "match"])
    p.add_argument(
        "--configs",
        default="configs/hybrid_supervised.toml,configs/unsup_v2.toml",
        help="target match: the two configs whose agreement is matched",
    )
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--games", default="")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--write", action="store_true", help="write configs/degradation/")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("log", help="show the experiment log")
    p.add_argument("-n", type=int, default=30)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("inspect", help="aggregate statistics of the data (never prints rows)")
    p.add_argument(
        "topic", choices=["ball", "players", "gaps", "tagging", "kickoffs", "labels", "regime"]
    )
    p.add_argument("--game", default=None, help="default: games.GAME")
    p.add_argument("--csv", help="submission CSV (topic labels)")
    p.add_argument(
        "--config", default="configs/hybrid_supervised.toml", help="labels for topic regime"
    )
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
