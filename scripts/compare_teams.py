"""Score our versions and two other teams' cached predictions on the same validation conditions.

    uv run python scripts/compare_teams.py measure label_free match_like idsse pff
    uv run python scripts/compare_teams.py report

The other teams' predictions on open data (Barca-Hackathon and Team Sigue, produced from their
public repositories) and the truths they were scored against sit in
`.cache/compete/{preds,truths}/`. Before it scores a condition, `measure` checks that the cached
truth is identical to ours. Both teams trained on Metrica and DFL/IDSSE, so on those sets their
scores are in-sample. PFF is the only fair set.

Prints aggregates only: means and matches won, per set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict

import numpy as np

from possession import config as config_mod
from possession import pipeline, stages, validation  # noqa: F401  (stages registers the methods)
from possession._paths import REPO_ROOT

COMPETE = REPO_ROOT / ".cache" / "compete"
SCORES = COMPETE / "scores"
OURS = {"V1": "hybrid_supervised", "V4": "c14_best", "V5": "c15_best"}
TEAMS = ("barca", "sigue")
FOLDERS = {
    "label_free": "metrica_label_free",
    "match_like": "metrica_match_like",
    "idsse": "idsse",
    "pff": "pff",
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(set_names: list[str]) -> None:
    """Score every condition of each set and write `.cache/compete/scores/<set>.json`."""
    grid = _script("grid14")
    memo = _script("decoder_grid14")._memoise_evidence(pipeline)
    configs = {k: config_mod.load(REPO_ROOT / "configs" / f"{v}.toml") for k, v in OURS.items()}
    SCORES.mkdir(parents=True, exist_ok=True)
    for set_name in set_names:
        folder = FOLDERS[set_name]
        rows, differ, missing = [], 0, 0
        for game, seed, match, truths in validation.conditions(**grid.set_kwargs(set_name)):
            memo.clear()
            key = f"{game}_s{seed}"
            path = COMPETE / "truths" / folder / f"{key}.npz"
            if not path.is_file():
                missing += 1
                continue
            cached = np.load(path)
            same = all(
                r in cached.files and np.array_equal(np.asarray(cached[r]), np.asarray(t))
                for r, t in truths.items()
            )
            differ += not same
            labels = {k: pipeline.run(c, match).labels for k, c in configs.items()}
            for team in TEAMS:
                pred = COMPETE / "preds" / team / folder / f"{key}.npz"
                labels[team] = np.load(pred)["labels"] if pred.is_file() else None
            for reading, truth in truths.items():
                row = {"game": str(game), "seed": int(seed), "reading": reading, "same": same}
                for k, v in labels.items():
                    row[k] = None if v is None else validation.score(v, truth)["spellF1"]
                rows.append(row)
        (SCORES / f"{set_name}.json").write_text(json.dumps(rows))
        print(f"{set_name}: {len(rows)} scores, {differ} truths differ, {missing} missing")


def report() -> None:
    """Mean spellF1 per set and version, and the matches V4 wins against each other team."""
    names = (*OURS, *TEAMS)
    print(f"{'set':<11}" + "".join(f"{n:>8}" for n in names) + "   V4 wins vs " + ", ".join(TEAMS))
    for set_name in FOLDERS:
        path = SCORES / f"{set_name}.json"
        if not path.is_file():
            continue
        rows = [r for r in json.loads(path.read_text()) if r["same"]]
        means = {}
        for n in names:
            vals = [r[n] for r in rows if r[n] is not None]
            means[n] = f"{np.mean(vals):8.3f}" if vals else f"{'-':>8}"
        wins = []
        for team in TEAMS:
            by_game = defaultdict(list)
            for r in rows:
                if r[team] is not None:
                    by_game[r["game"]].append(r["V4"] - r[team])
            won = sum(np.mean(d) > 0 for d in by_game.values())
            wins.append(f"{won}/{len(by_game)}")
        print(f"{set_name:<11}" + "".join(means[n] for n in names) + "   " + ", ".join(wins))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=["measure", "report"])
    parser.add_argument("sets", nargs="*", default=list(FOLDERS))
    args = parser.parse_args()
    measure(args.sets) if args.action == "measure" else report()


if __name__ == "__main__":
    main()
