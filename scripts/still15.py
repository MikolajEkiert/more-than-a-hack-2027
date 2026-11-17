"""Task 15: fit `match_regime_still`, i.e. `match_regime` plus a tracked ball lying still in play.

    uv run python scripts/still15.py calibrate [--write]

The competition match's tracked ball lies still (under 1 m/s) for half a second or more 15.5 times
per 10 graded minutes (`inspect_data.regime_vector`, label-free), against 7.1 under
`match_regime`. This searches `still_ball_rate` (events per 10 live minutes; `still_ball_s`
0.5-3 s, half of the events a spare ball by a touch line, the tracker noise of `match_regime`) for
the rate whose degraded Metrica matches (seed 0, the three games) show the match's still-run rate
and share of still tracked frames, read from the tracking alone. No label of the competition match
and no model output is used. Prints aggregates; `--write` writes the profile.
"""

import argparse
import json

import numpy as np

from possession import inspect_data, validation

BASE = "match_regime"
NAME = "match_regime_still"
RATES = (0.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0)
FIXED = {"still_ball_s": [0.5, 3.0], "still_ball_spare": 0.5}
TARGET_KEYS = ("still_runs_per_10min", "still_given_tracked")


def match_target() -> dict:
    from possession.loader import load_match

    vec = inspect_data.regime_vector(load_match(verbose=False))
    return {k: vec[k] for k in TARGET_KEYS}


def measure(rate: float) -> dict:
    rows = []
    for game in validation.GAMES:
        clean, events = validation._load(game)
        _, truth = validation.load_external(game, "touch")
        degraded = validation.degrade(
            clean, truth, seed=0, profile=BASE, still_ball_rate=rate, **FIXED
        )
        vec = inspect_data.regime_vector(degraded)
        rows.append({k: vec[k] for k in TARGET_KEYS})
    return {k: float(np.mean([r[k] for r in rows])) for k in TARGET_KEYS}


def calibrate(write: bool) -> None:
    target = match_target()
    print("competition match (tracking only):", target)
    grid = []
    for rate in RATES:
        stats = measure(rate)
        loss = sum(abs(stats[k] - target[k]) / max(abs(target[k]), 1e-9) for k in TARGET_KEYS)
        grid.append({"still_ball_rate": rate, **{k: round(v, 4) for k, v in stats.items()},
                     "loss": round(loss, 4)})  # fmt: skip
        print(grid[-1], flush=True)
    best = min(grid, key=lambda g: g["loss"])
    print("best:", best)
    if not write:
        return
    base = json.loads((validation.PROFILE_DIR / f"{BASE}.json").read_text())["params"]
    params = {**base, **FIXED, "still_ball_rate": best["still_ball_rate"]}
    document = {
        "name": NAME,
        "params": params,
        "source": (
            "match_regime (task 14) plus a tracked ball lying still while play goes on "
            "(task 15): the rate is fitted by scripts/still15.py so that the degraded Metrica "
            "matches show the still-run rate and the share of still tracked frames of the "
            "competition match's tracking (inspect_data.regime_vector, tracking only). No label "
            "and no published score."
        ),
        "evidence": {
            "target": target,
            "grid": grid,
            "fixed": FIXED,
            "games": list(validation.GAMES),
        },
    }
    path = validation.PROFILE_DIR / f"{NAME}.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["calibrate"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    calibrate(args.write)


if __name__ == "__main__":
    main()
