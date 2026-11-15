"""Task 15: the possession structure of the validation truths and of the competition match.

    uv run python scripts/composition15.py truths              # every set's truth, cached
    uv run python scripts/composition15.py match CONFIG ...    # the match's blocks, our own labels
    uv run python scripts/composition15.py report              # bin shares, weights, coverage

`truths` writes .cache/composition/truth.json: per group ("metrica", "idsse", "pff") and per game,
the descriptors of `possession.composition.window_descriptors` for every 10-minute window, from the
truth (the stoppages, hence the bins, are the same under every label reading; the readings shown are
`touch`, the provider's flags and `pff`). Game names are those of the cells of scripts/grid14.py, so
window i of a cell is window i here.

`match` writes .cache/composition/match.json: the share of the competition match's 10-minute blocks
in each stoppage-density bin, the mean over the configurations given (their labels, not the truth,
which nobody has), plus the means of the descriptors. Aggregates only; no single block is printed.
"""

import argparse
import json

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


def match(configs: list[str]) -> dict:
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
    (OUT / "match.json").write_text(json.dumps(out, indent=1))
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


def window_weights(set_name: str) -> dict:
    """{(game, window index): weight} for the windows of a set: the match's share of the window's
    stoppage-density bin over the set's own share of it."""
    group = load_truths()[group_of(set_name)]
    target = np.asarray(load_match()["bin_shares"])
    windows = [w for game in group.values() for w in game]
    weights = composition.importance(target, composition.bin_shares(windows))
    return {
        (game, i): float(weights[w["bin"]])
        for game, rows in group.items()
        for i, w in enumerate(rows)
    }


def report() -> None:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["truths", "match", "report"])
    parser.add_argument("configs", nargs="*")
    args = parser.parse_args()
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
        report()


if __name__ == "__main__":
    main()
