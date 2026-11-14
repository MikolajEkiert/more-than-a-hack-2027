"""Grids of hybrid variants behind two decisions of task 10 (docs/FUSION.md section 9).

Exploration only: it prints tables of aggregates, logs no run and writes no file. Every member is
compared with the submitted hybrid (model trained on IDSSE with raw labels, costs 5 and 12).

    uv run python scripts/variant_grid.py training label_free    # Metrica under one profile
    uv run python scripts/variant_grid.py training pff           # the 16 PFF test matches
    uv run python scripts/variant_grid.py family label_free match_like

training   the model trained on IDSSE, on IDSSE plus the 48 PFF training matches, or on those
           alone, each with a cost of 3, 5 and 8 for a change of team. Needs the PFF bundle in
           .cache/pff (src/possession/supervised.py says where it comes from).
family     the version 3 family of docs/FUSION.md section 8: training labels raw / tau1 / tau2,
           cost of a change of team 5 / 8 / 12, cost of a stoppage 12 / 20. With more than one
           match, the member is also chosen on all matches but one and scored on that one.

A target is the name of a degradation profile (the three Metrica matches under it) or `pff`.
"""

import sys
from multiprocessing import Pool

import numpy as np

from possession import config as config_mod
from possession import pipeline, stages, supervised, validation  # noqa: F401  stages registers

SUBMITTED = ("raw", "idsse", 5.0, 12.0)  # training labels, training data, turnover, stoppage
GRIDS = {
    "training": [
        ("raw", data, turn, 12.0)
        for data in ("idsse", "idsse+pff", "pff")
        for turn in (3.0, 5.0, 8.0)
    ],
    "family": [
        (labels, "idsse", turn, stop)
        for labels in ("raw", "tau1", "tau2")
        for turn in (5.0, 8.0, 12.0)
        for stop in (12.0, 20.0)
    ],
}


def config(member):
    labels, data, turn, stop = member
    source = {"method": "supervised", "labels": labels, "data": data}
    return config_mod.parse(
        {
            "name": "member",
            "ball_state": {"method": "blend", "sources": [source, "kickoff_wait"]},
            "team": {"method": "blend", "sources": [source]},
            "decode": {
                "method": "switch_cost",
                "turnover_cost": turn,
                "stop_cost": stop,
                "restart_cost": stop,
                "snap_start_s": 3.0,
                "snap_end_s": 3.0,
            },
        }
    )


def cells(job):
    """Every member of a grid on the conditions of one target (and one Metrica match)."""
    grid, target, game = job
    suite, profile = ("pff", None) if target == "pff" else ("metrica", target)
    games = None if game is None else (game,)
    out = []
    for name, seed, match, truths in validation.conditions(games, profile=profile, suite=suite):
        for member in GRIDS[grid]:
            labels = pipeline.run(config(member), match).labels
            for reading in validation.rule_readings(suite):
                scored = validation.score(labels, truths[reading])
                out.append(
                    {
                        "member": member,
                        "scenario": f"{target}/{reading}",
                        "game": name,
                        "seed": seed,
                        "spellF1": scored["spellF1"],
                        "team": scored["team_acc_both_alive"],
                        "ratio": scored["n_pred_possessions"] / scored["n_gt_possessions"],
                    }
                )
    return out


def mean(rows, member, key="spellF1", **where):
    """Mean of `key` over the cells of `member` whose fields lie in the given collections."""
    values = [
        row[key]
        for row in rows
        if row["member"] == member and all(row[field] in ok for field, ok in where.items())
    ]
    return float(np.mean(values))


def difference(rows, member, **where):
    return mean(rows, member, **where) - mean(rows, SUBMITTED, **where)


def table(rows, members):
    scenarios = sorted({row["scenario"] for row in rows})
    games = sorted({row["game"] for row in rows})
    print("spellF1 minus the submitted hybrid, by scenario: " + ", ".join(scenarios))
    for member in members:
        by_scenario = [difference(rows, member, scenario=[s]) for s in scenarios]
        won = sum(difference(rows, member, game=[g]) > 0 for g in games)
        print(
            f"{member!s:<34} {mean(rows, member):.3f}  "
            + "  ".join(f"{d:+.3f}" for d in by_scenario)
            + f"   mean {np.mean(by_scenario):+.3f}  worst {min(by_scenario):+.3f}"
            + f"  matches won {won}/{len(games)}"
            + f"  team accuracy {mean(rows, member, key='team'):.3f}"
            + f"  predicted/true possessions {mean(rows, member, key='ratio'):.2f}"
        )


def nested(rows, members):
    """Choose the member on all matches but one, score it on that one, average over the folds."""
    scenarios = sorted({row["scenario"] for row in rows})
    games = sorted({row["game"] for row in rows})
    candidates = [m for m in members if m != SUBMITTED]
    held_out = []
    for game in games:
        others = [g for g in games if g != game]
        on_others = {m: mean(rows, m, game=others) for m in candidates}
        pick = max(on_others, key=on_others.get)
        held_out.append([difference(rows, pick, game=[game], scenario=[s]) for s in scenarios])
        print(f"held out {game}: chosen on the other matches {pick}")
    by_scenario = np.mean(held_out, axis=0)
    print(
        "the chosen member on the match it was not chosen on, by scenario: "
        + "  ".join(f"{d:+.3f}" for d in by_scenario)
        + f"   mean {by_scenario.mean():+.3f}  worst {by_scenario.min():+.3f}"
    )
    on_all = {m: mean(rows, m) for m in candidates}
    print(f"best member on all matches: {max(on_all, key=on_all.get)}")


def main(argv):
    if len(argv) < 2 or argv[0] not in GRIDS:
        raise SystemExit(__doc__)
    grid, targets = argv[0], argv[1:]
    members = GRIDS[grid]
    for labels, data in sorted({member[:2] for member in members}):
        supervised.model(labels, data)  # train or load once, before the workers start
    jobs = []
    for target in targets:
        if target == "pff":
            jobs.append((grid, target, None))
        else:
            validation.degradation_profile(target)  # fail on an unknown name before any work
            jobs += [(grid, target, game) for game in validation.GAMES]
    with Pool(min(3, len(jobs))) as pool:
        rows = [row for part in pool.map(cells, jobs) for row in part]
    table(rows, members)
    if grid == "family" and len({row["game"] for row in rows}) > 1:
        nested(rows, members)


if __name__ == "__main__":
    main(sys.argv[1:])
