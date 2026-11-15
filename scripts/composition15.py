"""Task 15: the possession structure of the validation truths and of the competition match.

    uv run python scripts/composition15.py truths              # every set's truth, cached
    uv run python scripts/composition15.py match CONFIG ...    # the match's blocks, our own labels
    uv run python scripts/composition15.py report              # bin shares, weights, coverage
    uv run python scripts/composition15.py predicted CONFIG [--sets S,...]  # the instrument's view
    uv run python scripts/composition15.py report --instrument c14_best
    uv run python scripts/composition15.py lengths CONFIG ... [--sets S,...]  # stoppages by length

`truths` writes .cache/composition/truth.json: per group ("metrica", "idsse", "pff") and per game,
the descriptors of `possession.composition.window_descriptors` for every 10-minute window, from the
truth (the stoppages, hence the bins, are the same under every label reading; the readings shown are
`touch`, the provider's flags and `pff`). Game names are those of the cells of scripts/grid14.py, so
window i of a cell is window i here.

`match` writes .cache/composition/match.json: the share of the competition match's 10-minute blocks
in each stoppage-density bin, the mean over the configurations given (their labels, not the truth,
which nobody has), plus the means of the descriptors. Aggregates only; no single block is printed.

The instrument (added after task 15's step 4). Binning the match by our labels and the validation
windows by their truth compares two different measurements: version 4's labels count about 1.9 more
stoppages per 10 minutes than the truth under `match_regime_still`, the profile fitted to the
match's tracking. `predicted CONFIG` writes the descriptors of CONFIG's labels for every condition
of every set (.cache/composition/pred/<stem>/<set>.json) and the match's bin shares by CONFIG alone
(match_<stem>.json); `window_weights(set, instrument=stem)` then bins both sides by those labels: a
window's bin is the mean over seeds of its predicted density.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from possession import composition, validation
from possession._paths import CACHE_DIR, REPO_ROOT

OUT = CACHE_DIR / "composition"
GROUPS = ("metrica", "idsse", "pff")
FPS = 25


def group_of(set_name: str) -> str:
    """The truth group of a set name of scripts/grid14.py."""
    if set_name in ("idsse", "idsse_auto"):
        return "idsse"
    if set_name == "pff":
        return "pff"
    return "metrica"  # label_free, match_like and every metrica:<profile>


def truths() -> dict:
    from possession import supervised

    out = {group: {} for group in GROUPS}
    for game in validation.GAMES:
        match, labels = validation.load_external(game, "touch")
        out["metrica"][game] = composition.window_descriptors(labels, match.half, FPS)
    from solution import idsse

    for m in idsse.load_all():
        name = "idsse-" + str(m["name"]).split("_deg")[0].split("-")[-1].lower()
        out["idsse"][name] = composition.window_descriptors(
            np.asarray(m["label"]), np.asarray(m["half"]), FPS
        )
    for game_id, load in zip(
        supervised.pff_games()[1], supervised._pff_loaders(supervised.pff_games()[1]), strict=True
    ):
        m = load()
        out["pff"][f"pff-{game_id}"] = composition.window_descriptors(
            np.asarray(m["label"]), np.asarray(m["half"]), FPS
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truth.json").write_text(json.dumps(out))
    return out


def match(configs: list[str], out_path: Path | None = None) -> dict:
    from possession import config as config_mod
    from possession import pipeline, stages  # noqa: F401  (registers methods)
    from possession.loader import load_match

    game = load_match(verbose=False)
    shares, rows = [], []
    for path in configs:
        labels = pipeline.run(config_mod.load(path), game).labels
        windows = composition.window_descriptors(labels, game.half, game.fps)
        shares.append(composition.bin_shares(windows))
        rows += windows

    def mean(key):
        values = [w[key] for w in rows if w[key] is not None]
        return round(float(np.mean(values)), 4) if values else None

    out = {
        "configs": [str(p) for p in configs],
        "bin_shares": [round(float(v), 4) for v in np.mean(shares, axis=0)],
        "blocks_per_config": len(rows) // max(len(configs), 1),
        "mean_stoppages_per_10min": mean("stoppages_per_10min"),
        "mean_after_stoppage": mean("after_stoppage"),
        "mean_possessions_per_live_minute": mean("possessions_per_live_minute"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (out_path or OUT / "match.json").write_text(json.dumps(out, indent=1))
    return out


def load_truths() -> dict:
    path = OUT / "truth.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `scripts/composition15.py truths` first")
    return json.loads(path.read_text())


def load_match() -> dict:
    path = OUT / "match.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `scripts/composition15.py match CONFIG...`")
    return json.loads(path.read_text())


def window_weights(set_name: str, instrument: str | None = None) -> dict:
    """{(game, window index): weight} for the windows of a set: the match's share of the window's
    stoppage-density bin over the set's own share of it. Bins by the truth and the match's labels
    (task 15's step 2) unless `instrument` names a configuration measured by `predicted`, whose
    labels then bin both the match and the set's windows."""
    if instrument is not None:
        return _instrument_weights(set_name, instrument)
    group = load_truths()[group_of(set_name)]
    target = np.asarray(load_match()["bin_shares"])
    windows = [w for game in group.values() for w in game]
    weights = composition.importance(target, composition.bin_shares(windows))
    return {
        (game, i): float(weights[w["bin"]])
        for game, rows in group.items()
        for i, w in enumerate(rows)
    }


def pred_path(stem: str, set_name: str):
    return OUT / "pred" / stem / f"{set_name.replace(':', '_')}.json"


def predicted_bins(set_name: str, instrument: str) -> dict:
    """{(game, window index): bin} of a set's windows by the instrument's labels (mean density over
    the seeds of the condition)."""
    path = pred_path(instrument, set_name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `composition15.py predicted` first")
    density: dict[tuple, list[float]] = {}
    for row in json.loads(path.read_text()):
        for i, w in enumerate(row["windows"]):
            density.setdefault((row["game"], i), []).append(w["stoppages_per_10min"])
    return {k: int(np.digitize(np.mean(v), composition.STOP_BINS)) for k, v in density.items()}


def _instrument_weights(set_name: str, instrument: str) -> dict:
    target = np.asarray(json.loads((OUT / f"match_{instrument}.json").read_text())["bin_shares"])
    bins = predicted_bins(set_name, instrument)
    shares = np.bincount(list(bins.values()), minlength=composition.N_BINS) / max(len(bins), 1)
    weights = composition.importance(target, shares)
    return {key: float(weights[b]) for key, b in bins.items()}


def _halves(half: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def stoppage_errors(labels, truth, half: np.ndarray, fps: int) -> dict:
    """Counts behind a configuration's stoppage bias on a labelled match: graded minutes, true and
    predicted stoppages, predicted ones lying mostly in true play (spurious), extra predicted ones
    inside one true stoppage (splits), true stoppages no predicted one covers (missed)."""
    import possession_metrics as pm

    pc, tc = composition._codes(labels), composition._codes(truth)
    out = {k: 0.0 for k in ("minutes", "true", "pred", "spurious", "split_extra", "missed")}
    lengths = []
    for a0, b0 in _halves(np.asarray(half)):
        out["minutes"] += (b0 - a0) / fps / 60
        ps = [(a, b) for a, b, k in pm.segments(pc, [(a0, b0)]) if k == pm.D]
        ts = [(a, b) for a, b, k in pm.segments(tc, [(a0, b0)]) if k == pm.D]
        out["true"] += len(ts)
        out["pred"] += len(ps)
        dead = np.zeros(len(tc), dtype=bool)
        for a, b in ts:
            dead[a:b] = True
        for a, b in ps:
            if dead[a:b].mean() < 0.5:
                out["spurious"] += 1
                lengths.append((b - a) / fps)
        for a, b in ts:
            n = sum(1 for c, d in ps if c < b and d > a and dead[c:d].mean() >= 0.5)
            out["missed"] += n == 0
            out["split_extra"] += max(n - 1, 0)
    out["spurious_lengths"] = lengths
    return out


LENGTH_EDGES = (2.0, 4.0, 8.0, 15.0, 30.0)  # seconds; bins <2, 2-4, 4-8, 8-15, 15-30, >=30


def stoppage_lengths(labels, half: np.ndarray, fps: int) -> tuple[np.ndarray, float]:
    """Counts of stoppages (D spells inside a half) per length bin (`LENGTH_EDGES`), and the
    graded minutes they were counted over."""
    import possession_metrics as pm

    codes = composition._codes(labels)
    lengths, minutes = [], 0.0
    for a0, b0 in _halves(np.asarray(half)):
        minutes += (b0 - a0) / fps / 60
        lengths += [(b - a) / fps for a, b, k in pm.segments(codes, [(a0, b0)]) if k == pm.D]
    counts = np.bincount(np.digitize(lengths, LENGTH_EDGES), minlength=len(LENGTH_EDGES) + 1)
    return counts.astype(np.float64), minutes


def lengths(configs: list[str], sets: list[str]) -> None:
    """Stoppages per 10 graded minutes by length: the match by each config's labels, and every set
    by the truth and by each config's labels (aggregates only)."""
    from possession import config as config_mod
    from possession import pipeline, stages  # noqa: F401  (registers methods)
    from possession.loader import load_match

    head = ["<2", "2-4", "4-8", "8-15", "15-30", ">=30"]
    print(f"{'stoppages per 10 min':<44}" + "".join(f"{h:>7}" for h in head) + "    all")

    def show(label, counts, minutes):
        per10 = counts / minutes * 10
        print(f"{label:<44}" + "".join(f"{v:>7.2f}" for v in per10) + f"{per10.sum():>7.2f}")

    cfgs = {Path(c).stem: config_mod.load(c) for c in configs}
    game = load_match(verbose=False)
    for stem, cfg in cfgs.items():
        show(
            f"match, {stem}", *stoppage_lengths(pipeline.run(cfg, game).labels, game.half, game.fps)
        )
    grid14 = _grid14()
    for set_name in sets:
        acc = {"truth": [np.zeros(6), 0.0], **{s: [np.zeros(6), 0.0] for s in cfgs}}
        for _, _, match, truths in validation.conditions(**grid14.set_kwargs(set_name)):
            pairs = [("truth", truths[sorted(truths)[0]])]
            pairs += [(s, pipeline.run(c, match).labels) for s, c in cfgs.items()]
            for key, labels in pairs:
                counts, minutes = stoppage_lengths(labels, match.half, match.fps)
                acc[key][0] += counts
                acc[key][1] += minutes
        for key, (counts, minutes) in acc.items():
            show(f"{set_name}, {key}", counts, minutes)


def predicted_one(job) -> tuple[str, dict, float]:
    """One set: the instrument's window descriptors per condition, and its stoppage errors."""
    config, set_name = job
    from possession import config as config_mod
    from possession import pipeline, stages  # noqa: F401  (registers methods)

    grid14 = _grid14()
    cfg = config_mod.load(config)
    rows, total, lengths = [], {}, []
    for game, seed, match, truths in validation.conditions(**grid14.set_kwargs(set_name)):
        labels = pipeline.run(cfg, match).labels
        rows.append(
            {
                "game": str(game),
                "seed": seed,
                "windows": composition.window_descriptors(labels, match.half, match.fps),
            }
        )
        errors = stoppage_errors(labels, truths[sorted(truths)[0]], match.half, match.fps)
        lengths += errors.pop("spurious_lengths")
        for k, v in errors.items():
            total[k] = total.get(k, 0.0) + v
    stem = Path(config).stem
    path = pred_path(stem, set_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    per10 = {k: round(v / total["minutes"] * 10, 2) for k, v in total.items() if k != "minutes"}
    return set_name, per10, float(np.median(lengths)) if lengths else 0.0


def predicted(config: str, sets: list[str], jobs: int) -> None:
    """The instrument's descriptors on every set (cached) and on the match (match_<stem>.json)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    stem = Path(config).stem
    todo = [(config, s) for s in sets if not pred_path(stem, s).is_file()]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        for future in as_completed([pool.submit(predicted_one, job) for job in todo]):
            set_name, per10, median = future.result()
            print(f"{stem} {set_name:<28} per 10 min {per10}  spurious median {median:.1f} s")
    out = match([config], OUT / f"match_{stem}.json")
    print(f"match by {stem}: bin shares {out['bin_shares']}, "
          f"{out['mean_stoppages_per_10min']} stoppages per 10 min")  # fmt: skip


def _grid14():
    import importlib.util

    spec = importlib.util.spec_from_file_location("grid14", REPO_ROOT / "scripts" / "grid14.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(instrument: str | None = None) -> None:
    if instrument is not None:
        return report_instrument(instrument)
    target = np.asarray(load_match()["bin_shares"])
    meta = load_match()
    print(
        f"competition match (our labels, {meta['blocks_per_config']} blocks per configuration): "
        f"bin shares {np.round(target, 3).tolist()} for stoppages per 10 min "
        f"<{composition.STOP_BINS[0]:g} / {composition.STOP_BINS[0]:g}-{composition.STOP_BINS[1]:g}"
        f" / {composition.STOP_BINS[1]:g}-{composition.STOP_BINS[2]:g} / "
        f">={composition.STOP_BINS[2]:g}; mean {meta['mean_stoppages_per_10min']} per 10 min, "
        f"after a stoppage {meta['mean_after_stoppage']}"
    )
    for group, games in load_truths().items():
        windows = [w for rows in games.values() for w in rows]
        shares = composition.bin_shares(windows)
        weights = composition.importance(target, shares)
        print(
            f"{group:<8} {len(windows):>4} windows  shares {np.round(shares, 3).tolist()}  "
            f"weights {np.round(weights, 2).tolist()}  match covered "
            f"{composition.coverage(target, shares):.2f}"
        )


def report_instrument(instrument: str) -> None:
    """Bin shares of the match and of every set's windows, all by the instrument's labels."""
    meta = json.loads((OUT / f"match_{instrument}.json").read_text())
    target = np.asarray(meta["bin_shares"])
    print(
        f"competition match by {instrument}: bin shares {np.round(target, 3).tolist()}; mean "
        f"{meta['mean_stoppages_per_10min']} stoppages per 10 min, after a stoppage "
        f"{meta['mean_after_stoppage']}"
    )
    for path in sorted((OUT / "pred" / instrument).glob("*.json")):
        rows = json.loads(path.read_text())
        windows = [w for row in rows for w in row["windows"]]
        density = np.mean([w["stoppages_per_10min"] for w in windows])
        after = np.mean([w["after_stoppage"] for w in windows if w["after_stoppage"] is not None])
        bins = predicted_bins(path.stem.replace("metrica_", "metrica:", 1), instrument)
        shares = np.bincount(list(bins.values()), minlength=composition.N_BINS) / len(bins)
        print(
            f"{path.stem:<28} {len(bins):>4} windows  {density:5.2f} per 10 min, after a stoppage "
            f"{after:.3f}  shares {np.round(shares, 3).tolist()}  weights "
            f"{np.round(composition.importance(target, shares), 2).tolist()}  covered "
            f"{composition.coverage(target, shares):.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["truths", "match", "report", "predicted", "lengths"])
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--sets", default="")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--instrument", default=None, help="report: bin by this config's labels")
    args = parser.parse_args()
    sets = [s for s in args.sets.split(",") if s]
    if args.action == "predicted":
        for config in args.configs:
            predicted(config, sets, args.jobs)
        return
    if args.action == "lengths":
        lengths(args.configs, sets)
        return
    if args.action == "truths":
        out = truths()
        print({group: len(games) for group, games in out.items()})
    elif args.action == "match":
        configs = args.configs or [
            str(REPO_ROOT / "configs" / "c14_best.toml"),
            str(REPO_ROOT / "configs" / "hybrid_supervised.toml"),
        ]
        print(json.dumps(match(configs), indent=1))
    else:
        report(args.instrument)


if __name__ == "__main__":
    main()
