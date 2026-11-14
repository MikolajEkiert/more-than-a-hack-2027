"""Task 14, decoder options: measure many configurations that differ mostly in [decode].

    uv run python scripts/decoder_grid14.py CONFIG [CONFIG ...] [--sets S,...] [--jobs N] [--force]

The fast path of `scripts/grid14.py measure` for a family of decoder variants. Every
configuration still goes through `pipeline.run`, unchanged, but within one condition the output of
the ball_state and team stages is memoised by stage configuration, so configurations sharing those
two stages compute the evidence (the expensive part, about 13 s per condition against about 1 s
for the decoder) once. The cells are the same files with the same content as grid14.py writes
(.cache/family/<config stem>/<set>.json and config.txt), so `grid14.py compare|decide|decide-b`
read them unchanged. One job per set; each prints, per configuration, the mean spellF1 and the
mean share of graded frames whose label differs from the first configuration given (a check that
an option fires at all). Aggregates of other matches only.
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from possession._paths import REPO_ROOT


def _grid14():
    spec = importlib.util.spec_from_file_location("grid14", REPO_ROOT / "scripts" / "grid14.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _grid14()


def _memoise_evidence(pipeline) -> dict:
    """Make `pipeline._call` keep the ball_state and team outputs of the current condition."""
    memo: dict = {}
    original = pipeline._call

    def call(stage, stage_config, *inputs):
        if stage == "decode":
            return original(stage, stage_config, *inputs)
        params = json.dumps(stage_config.params, sort_keys=True)
        key = (stage, stage_config.method, params)
        if key not in memo:
            memo[key] = original(stage, stage_config, *inputs)
        return memo[key]

    pipeline._call = call
    return memo


def measure_set(job) -> list[tuple[str, str, float, float]]:
    """Score every configuration on every condition of one set, sharing the evidence; write the
    cells of those in `write` (never those of grid14's aliases V1, M3 to M6)."""
    set_name, configs, write = job
    from possession import config as config_mod
    from possession import pipeline, stages, validation  # noqa: F401  (registers methods)

    memo = _memoise_evidence(pipeline)
    loaded = [(name, config_mod.load(path)) for name, path in configs]
    kw = G.set_kwargs(set_name)
    readings = validation.rule_readings(kw["suite"])
    cells = {name: [] for name, _ in loaded}
    differs = {name: [] for name, _ in loaded}
    for game, seed, match, truths in validation.conditions(**kw):
        memo.clear()  # evidence belongs to one condition
        reference = None
        for name, cfg in loaded:
            labels = pipeline.run(cfg, match).labels
            if reference is None:
                reference = labels
            differs[name].append(float(np.mean(labels[match.graded] != reference[match.graded])))
            for reading in readings:
                truth = truths[reading]
                cells[name].append(
                    {
                        "game": str(game),
                        "seed": seed,
                        "reading": reading,
                        **validation.score(labels, truth),
                        "windows": validation.window_scores(labels, truth, match.half, match.fps),
                    }
                )
    out = []
    for name, path in configs:
        if name in write and name not in G.ALIASES:
            target = G.cell_path(name, set_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(cells[name]))
            tmp.replace(target)
            (G.OUT / name / "config.txt").write_text(str(Path(path).resolve()))
        mean = float(np.mean([c["spellF1"] for c in cells[name]]))
        out.append((name, set_name, mean, float(np.mean(differs[name]))))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--sets", default="label_free,match_like,pff")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = [(G.name_of(c), str(Path(c).resolve())) for c in args.configs]
    jobs = []
    for set_name in [s for s in args.sets.split(",") if s]:
        todo = [
            (name, path)
            for name, path in configs
            if name not in G.ALIASES
            and (
                args.force
                or not G.has(name, set_name)
                or "windows" not in G.load(name, set_name)[0]
            )
        ]
        if todo:
            # the first configuration given is the reference of the "differs" column
            run = configs[:1] + [c for c in todo if c != configs[0]]
            jobs.append((set_name, run, {name for name, _ in todo}))
    if not jobs:
        print("nothing to measure (cached)")
        return
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(measure_set, job) for job in jobs]
        for future in as_completed(futures):
            for name, set_name, value, differ in future.result():
                print(f"{name:<34} {set_name:<12} {value:.4f}  differs {differ:.4f}", flush=True)


if __name__ == "__main__":
    main()
