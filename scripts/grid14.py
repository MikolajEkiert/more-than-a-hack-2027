"""Task 14: measure any configuration on every validation set, compare it with version 1, decide.

    uv run python scripts/grid14.py measure CONFIG [CONFIG ...] [--sets S,...] [--jobs N] [--force]
    uv run python scripts/grid14.py compare NAME [NAME ...] [--sets S,...] [--against V1]
    uv run python scripts/grid14.py decide NAME [NAME ...]        # rule A (docs/FUSION.md s. 10)
    uv run python scripts/grid14.py decide-b NAME [NAME ...] [--evidence P,...]   # rule B (task 14)
    uv run python scripts/grid14.py footprint CONFIG [CONFIG ...]

A member's name is its config's file stem, except the members of scripts/family_grid.py (V1 is
configs/hybrid_supervised.toml, M3 to M6), which keep their names. Cells, one per condition and
counted label reading, with the spellF1 of every 10-minute window, are cached in
.cache/family/<name>/<set>.json, the files family_grid.py reads, so both scripts share them.

Sets: label_free, match_like, idsse, idsse_auto, pff, and metrica:<profile> for any other profile
in configs/degradation/. Aggregates of other matches only; of the competition match only the
label-free footprint.
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from possession._paths import CACHE_DIR, REPO_ROOT

OUT = CACHE_DIR / "family"
ALIASES = {
    "V1": "configs/hybrid_supervised.toml",
    "M3": "configs/hybrid_regime_mix.toml",
    "M4": "configs/hybrid_tta.toml",
    "M5": "configs/hybrid_sequence_blend.toml",
    "M6": "configs/hybrid_sequence.toml",
}
BASE_SETS = {
    "label_free": {"suite": "metrica", "profile": "label_free"},
    "match_like": {"suite": "metrica", "profile": "match_like"},
    "idsse": {"suite": "idsse"},
    "idsse_auto": {"suite": "idsse_auto"},
    "pff": {"suite": "pff"},
}
N_BOOT = 2000
FOOTPRINT = {"possessions_per_live_minute": (3.5, 7.0), "dead_share": (0.35, 0.50)}


def set_kwargs(set_name: str) -> dict:
    if set_name in BASE_SETS:
        return dict(BASE_SETS[set_name])
    if set_name.startswith("metrica:"):
        return {"suite": "metrica", "profile": set_name.split(":", 1)[1]}
    raise KeyError(f"unknown set {set_name!r}")


def suite_of(set_name: str) -> str:
    return set_kwargs(set_name)["suite"]


def name_of(config: str) -> str:
    path = Path(config).resolve()
    for alias, target in ALIASES.items():
        if path == (REPO_ROOT / target).resolve():
            return alias
    return path.stem


def config_of(name: str) -> Path:
    if name in ALIASES:
        return REPO_ROOT / ALIASES[name]
    meta = OUT / name / "config.txt"
    if meta.is_file():
        return Path(meta.read_text().strip())
    return REPO_ROOT / "configs" / f"{name}.toml"


def cell_path(name: str, set_name: str) -> Path:
    return OUT / name / f"{set_name.replace(':', '@')}.json"


def load(name: str, set_name: str) -> list[dict]:
    return json.loads(cell_path(name, set_name).read_text())


def has(name: str, set_name: str) -> bool:
    return cell_path(name, set_name).is_file()


# -- measuring ---------------------------------------------------------------------------------


def measure_one(job) -> tuple[str, str, float]:
    """Score one configuration on every condition of one set; cells cached as family_grid's."""
    name, config, set_name = job
    from possession import config as config_mod
    from possession import pipeline, stages, validation  # noqa: F401  (registers methods)

    cfg = config_mod.load(config)
    kw = set_kwargs(set_name)
    readings = validation.rule_readings(kw["suite"])
    cells = []
    for game, seed, match, truths in validation.conditions(**kw):
        labels = pipeline.run(cfg, match).labels
        for reading in readings:
            truth = truths[reading]
            cells.append(
                {
                    "game": str(game),
                    "seed": seed,
                    "reading": reading,
                    **validation.score(labels, truth),
                    "windows": validation.window_scores(labels, truth, match.half, match.fps),
                }
            )
    path = cell_path(name, set_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cells))
    tmp.replace(path)
    (OUT / name / "config.txt").write_text(str(Path(config).resolve()))
    return name, set_name, float(np.mean([c["spellF1"] for c in cells]))


def measure(configs: list[str], sets: list[str], jobs: int, force: bool) -> None:
    todo = []
    for config in configs:
        name = name_of(config)
        for set_name in sets:
            path = cell_path(name, set_name)
            if path.is_file() and not force and "windows" in load(name, set_name)[0]:
                continue
            todo.append((name, str(Path(config).resolve()), set_name))
    if not todo:
        print("nothing to measure (cached)")
        return
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(measure_one, job) for job in todo]
        for future in as_completed(futures):
            name, set_name, value = future.result()
            print(f"{name:<34} {set_name:<22} {value:.4f}", flush=True)


# -- comparing ---------------------------------------------------------------------------------


def _key(cell) -> tuple:
    return (cell["game"], cell["seed"], cell["reading"])


def paired(a: list[dict], b: list[dict], games=None) -> dict:
    """Paired differences a - b over the conditions both have (optionally only `games`)."""
    ka, kb = {_key(c): c for c in a}, {_key(c): c for c in b}
    common = sorted(k for k in set(ka) & set(kb) if games is None or k[0] in games)
    if not common:
        return {}
    diff = {k: ka[k]["spellF1"] - kb[k]["spellF1"] for k in common}
    readings = sorted({k[2] for k in common})
    names = sorted({k[0] for k in common})
    by_reading = {r: float(np.mean([d for k, d in diff.items() if k[2] == r])) for r in readings}
    by_game = {g: float(np.mean([d for k, d in diff.items() if k[0] == g])) for g in names}
    # (game, window) blocks: mean over seeds and readings of the window differences
    blocks: dict[tuple, list[float]] = {}
    for k in common:
        wa, wb = ka[k].get("windows"), kb[k].get("windows")
        if not wa or not wb:
            continue
        for i, (x, y) in enumerate(zip(wa, wb, strict=True)):
            if x is not None and y is not None:
                blocks.setdefault((k[0], i), []).append(x - y)
    interval = None
    if blocks:
        values = np.array([np.mean(v) for v in blocks.values()])
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
        means = values[draws].mean(axis=1)
        interval = (float(np.percentile(means, 5)), float(np.percentile(means, 95)))
    return {
        "mean": float(np.mean(list(diff.values()))),
        "by_reading": by_reading,
        "by_game": by_game,
        "won": sum(v > 0 for v in by_game.values()),
        "n_games": len(by_game),
        "interval": interval,
        "a": float(np.mean([ka[k]["spellF1"] for k in common])),
        "b": float(np.mean([kb[k]["spellF1"] for k in common])),
    }


def compare(names: list[str], sets: list[str], against: str) -> None:
    for set_name in sets:
        if not has(against, set_name):
            print(f"{set_name}: no cells for {against}")
            continue
        ref = load(against, set_name)
        print(f"\n[{set_name}] {against} = {np.mean([c['spellF1'] for c in ref]):.4f}")
        for name in names:
            if not has(name, set_name):
                print(f"  {name:<32} (not measured)")
                continue
            p = paired(load(name, set_name), ref)
            iv = p["interval"]
            iv_text = f"[{iv[0]:+.4f}, {iv[1]:+.4f}]" if iv else "[n/a]"
            reading_text = "  ".join(f"{r} {v:+.4f}" for r, v in p["by_reading"].items())
            print(
                f"  {name:<32} {p['a']:.4f}  diff {p['mean']:+.4f} {iv_text}  {reading_text}"
                f"  won {p['won']}/{p['n_games']}"
            )


# -- rule A: the team's rule (docs/FUSION.md section 10) ---------------------------------------


def decide_a(names: list[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "family_grid", REPO_ROOT / "scripts" / "family_grid.py"
    )
    family_grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family_grid)
    for name in names:
        family_grid.MEMBERS[name] = str(config_of(name))
    family_grid.decide(names)


# -- rule B: matched to the match's ball regime (tasks/14-own-directions.md) --------------------


def decide_b(names: list[str], evidence: list[str]) -> None:
    games = sorted({c["game"] for c in load("V1", evidence[0])})

    def mean_on(name, chosen):
        values = [
            c["spellF1"] for p in evidence for c in load(name, p) if c["game"] in chosen
        ]  # fmt: skip
        return float(np.mean(values))

    held, block_values, cell_diffs = {}, [], {}
    for g in games:
        others = [x for x in games if x != g]
        pick = max(names, key=lambda m: mean_on(m, others))
        held[g] = pick
        blocks: dict[int, list[float]] = {}
        for p in evidence:
            res = paired(load(pick, p), load("V1", p), games={g})
            for r, v in res["by_reading"].items():
                cell_diffs.setdefault((p, r), []).append(v)
            a = {_key(c): c for c in load(pick, p) if c["game"] == g}
            b = {_key(c): c for c in load("V1", p) if c["game"] == g}
            for k in sorted(set(a) & set(b)):
                for i, (x, y) in enumerate(zip(a[k]["windows"], b[k]["windows"], strict=True)):
                    if x is not None and y is not None:
                        blocks.setdefault(i, []).append(x - y)
        block_values += [float(np.mean(v)) for v in blocks.values()]
    by = {k: float(np.mean(v)) for k, v in cell_diffs.items()}
    gain = float(np.mean(list(by.values())))
    values = np.array(block_values)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    low = float(np.percentile(means, 5))
    print("held-out picks:", held)
    print(
        "held-out difference by (profile, reading):",
        {f"{p}/{r}": round(v, 4) for (p, r), v in by.items()},
    )
    print(f"held-out mean over the evidence {gain:+.4f}; 5% bootstrap bound {low:+.4f}")
    ok = gain >= 0.010 and low > 0 and all(v >= -0.010 for v in by.values())
    final = max(names, key=lambda m: mean_on(m, games))
    print(f"evidence passes: {ok}; member chosen on all three matches: {final}")
    checks = {"pff": (0.0, 9), "idsse_auto": (0.0, None), "idsse": (-0.010, None)}
    for set_name, (floor, need) in checks.items():
        if not (has(final, set_name) and has("V1", set_name)):
            print(f"{set_name}: not measured -> FAIL")
            ok = False
            continue
        res = paired(load(final, set_name), load("V1", set_name))
        passes = res["mean"] >= floor and (need is None or res["won"] >= need)
        ok = ok and passes
        print(
            f"{set_name}: {res['mean']:+.4f}, won {res['won']}/{res['n_games']} "
            f"-> {'passes' if passes else 'FAIL'}"
        )
    fp = footprint_of(config_of(final))
    inside = all(lo <= fp[k] <= hi for k, (lo, hi) in FOOTPRINT.items())
    print("footprint:", {k: fp[k] for k in FOOTPRINT}, "->", "inside" if inside else "OUTSIDE")
    ok = ok and inside
    print("VERDICT (rule B):", f"change to {final}" if ok else "version 1 stays")


# -- footprint on the competition match (label-free aggregates) ---------------------------------


def footprint_of(config: Path) -> dict:
    from possession import config as config_mod
    from possession import loader, pipeline, scoring

    match = loader.load_match(verbose=False)
    labels = pipeline.run(config_mod.load(config), match).labels
    return scoring.footprint(labels, match.half, match.fps)


def footprints(configs: list[str]) -> None:
    import possession_metrics as pm
    from possession import config as config_mod
    from possession import loader, pipeline, scoring

    match = loader.load_match(verbose=False)
    ref = pipeline.run(config_mod.load(config_of("V1")), match).labels
    gt = pm.label_codes(np.where(match.half > 0, ref, "X"))
    for config in configs:
        labels = pipeline.run(config_mod.load(config), match).labels
        fp = scoring.footprint(labels, match.half, match.fps)
        pred = pm.label_codes(labels)
        runs = pm.valid_runs(gt)
        agreement = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
        differ = float(np.mean(labels[match.half > 0] != ref[match.half > 0]))
        print(
            f"{name_of(config):<34} possessions {fp['possessions']:>4}  "
            f"per live min {fp['possessions_per_live_minute']:.2f}  "
            f"dead {fp['dead_share']:.4f}  mean {fp['mean_possession_s']:.1f} s  "
            f"spellF1 vs V1 {agreement:.3f}  frames differing {differ:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["measure", "compare", "decide", "decide-b", "footprint"])
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sets", default=",".join(BASE_SETS))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--against", default="V1")
    parser.add_argument("--evidence", default="label_free,match_like")
    args = parser.parse_args()
    sets = [s for s in args.sets.split(",") if s]
    if args.action == "measure":
        measure(args.items, sets, args.jobs, args.force)
    elif args.action == "compare":
        compare(args.items, sets, args.against)
    elif args.action == "decide":
        decide_a(args.items)
    elif args.action == "decide-b":
        decide_b(args.items, [s for s in args.evidence.split(",") if s])
    else:
        footprints(args.items)


if __name__ == "__main__":
    main()
