"""Task 14: how far each validation set resembles the competition match, statistic by statistic.

    uv run python scripts/regime_compare.py measure [SET ...] [--jobs N] [--force]
    uv run python scripts/regime_compare.py table [SET ...] [--truth]
    uv run python scripts/regime_compare.py fit [--jobs N] [--write]

SET is `match` (the competition match, version 1's labels, no truth), label_free, match_like,
metrica:<profile> (any profile in configs/degradation/), idsse, idsse_auto or pff. Default: all.

For every condition of a set the same vector is computed (`inspect_data.regime_vector`): from the
tracking alone (ball missing, gap lengths, where a tracked ball is, how it reappears, running
players), and split by version 1's labels (P(ball missing | play or stoppage), stoppage lengths,
play without a ball by kind, where inside stoppages the ball is in view, possession density, dead
share), plus version 1's team margin and its agreement with unsup_v2 (`validation.agreement`).
On the labelled sets the label-based part is also computed with the truth. Version 1's labels are
what the match can be described with; the truth columns show how far they distort the picture.

`table` prints the set means and a similarity table: per aspect, the mean relative distance of
each set to the match (0 = the same; `relative`), and which set is closest. Aggregates only; the
cache is .cache/regime_compare/<set>.json (means of per-condition aggregates).

`fit` fits the degradation profile `match_regime` to the match's label-free statistics: the same
vector under version 1's labels on the degraded Metrica matches (seed 0), plus the agreement of
version 1 with unsup_v2 (`validation.calibrate_to_match`), in two grids: first how the ball is
seen in stoppages, then how much of play loses it and how far the ball is off its possessor.
`--write` writes configs/degradation/match_regime.json with the fit as its evidence.
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from possession._paths import CACHE_DIR, REPO_ROOT

OUT = CACHE_DIR / "regime_compare"
V1 = REPO_ROOT / "configs" / "hybrid_supervised.toml"
OTHER = REPO_ROOT / "configs" / "unsup_v2.toml"
SETS = ("match", "label_free", "match_like", "idsse", "idsse_auto", "pff")
SHORT = {"match": "match", "label_free": "lf", "match_like": "ml", "idsse": "idsse",
         "idsse_auto": "id_auto", "pff": "pff", "metrica:match_regime": "m_regime"}  # fmt: skip

# Aspects of the similarity table: which statistics describe what.
ASPECTS = {
    "ball lost in play": (
        "missing_given_live", "live_missing_free", "live_missing_lead_in",
        "live_missing_lead_out", "live_gaps_per_10min_live", "live_gap_s_p50", "live_gap_s_p90",
        "free_gap_s_p50", "lead_in_s_p50", "live_dropouts_per_10min_live",
    ),
    "ball lost in stoppages": (
        "missing_given_dead", "dead_tracked_first_third", "dead_tracked_middle_third",
        "dead_tracked_last_third", "stoppage_any_tracked", "stoppage_start_tracked",
        "stoppage_end_tracked", "dead_visible_per_stoppage", "dead_visible_s_p50",
        "dead_visible_s_p90",
    ),
    "ball in view in stoppages": (
        "dead_tracked_still", "dead_tracked_slow", "dead_tracked_moving", "dead_tracked_outside",
        "stoppage_end_still", "end_visible_s_p50", "running_given_dead_tracked",
    ),
    "gaps (tracking only)": (
        "ball_missing", "gaps_under_1s_per_10min", "gaps_1_3s_per_10min",
        "gaps_over_3s_per_10min", "long_gap_s_p25", "long_gap_s_p50", "long_gap_s_p75",
        "long_gap_s_p90",
    ),
    "tracked ball (tracking only)": (
        "outside_given_tracked", "still_given_tracked", "moving_given_tracked",
        "still_runs_per_10min", "still_run_s_p50", "reappear_speed_p50", "reappear_moving",
        "reappear_still", "reappear_outside",
    ),
    "ball near a player": (
        "nearest_player_m_p25", "nearest_player_m_p50", "nearest_player_m_p75",
        "nearest_player_m_p90", "nearest_player_live_m_p50", "nearest_player_live_m_p90",
    ),
    "players running": (
        "running_given_missing", "running_given_tracked", "running_given_live",
        "running_given_dead_missing",
    ),
    "stoppages": (
        "dead_share", "stoppages_per_10min", "stoppage_s_p25", "stoppage_s_p50",
        "stoppage_s_p75", "stoppage_s_p90", "short_stoppage_share",
    ),
    "possessions": ("possessions_per_live_minute",),
    "hardness (no labels)": ("team_margin", "possession_agreement", "team_agreement"),
}  # fmt: skip


def _grid14():
    spec = importlib.util.spec_from_file_location("grid14", REPO_ROOT / "scripts" / "grid14.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_path(set_name: str):
    return OUT / f"{set_name.replace(':', '@')}.json"


# -- measuring ---------------------------------------------------------------------------------


def _row(match, v1, other, truths=None, readings=()) -> dict:
    from possession import inspect_data, pipeline, scoring, validation

    a, b = pipeline.run(v1, match), pipeline.run(other, match)
    agreement = validation.agreement(match, a, b)
    row = {
        "v1": {
            **inspect_data.regime_vector(match, a.labels),
            **{k: agreement[k] for k in validation.AGREEMENT_KEYS},
        }
    }
    if truths:
        truth = np.asarray(truths[readings[0]]).astype("<U1")  # stoppages: the same in all
        row["truth"] = inspect_data.regime_vector(match, truth)
        for reading in readings:
            fp = scoring.footprint(np.asarray(truths[reading]).astype("<U1"), match.half, match.fps)
            row["truth"][f"possessions_per_live_minute@{reading}"] = fp[
                "possessions_per_live_minute"
            ]
    return row


def measure_set(set_name: str) -> tuple[str, int]:
    """Every condition of one set; writes the per-condition vectors (aggregates) to the cache."""
    from possession import config as config_mod
    from possession import loader, stages, validation  # noqa: F401  (registers methods)

    v1, other = config_mod.load(V1), config_mod.load(OTHER)
    rows = []
    if set_name == "match":
        match = loader.load_match(verbose=False)
        rows.append({"game": match.game, "seed": None, **_row(match, v1, other)})
    else:
        kw = _grid14().set_kwargs(set_name)
        readings = validation.rule_readings(kw["suite"])
        for game, seed, match, truths in validation.conditions(**kw):
            rows.append(
                {"game": str(game), "seed": seed, **_row(match, v1, other, truths, readings)}
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = cache_path(set_name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)
    return set_name, len(rows)


def measure(sets: list[str], jobs: int, force: bool) -> None:
    todo = [s for s in sets if force or not cache_path(s).is_file()]
    if not todo:
        print("nothing to measure (cached)")
        return
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(measure_set, s) for s in todo]
        for future in as_completed(futures):
            name, count = future.result()
            print(f"{name}: {count} conditions", flush=True)


# -- the table ---------------------------------------------------------------------------------


def set_means(set_name: str, part: str = "v1") -> dict:
    """Mean over the set's conditions of every statistic (None where no condition has it)."""
    rows = [r[part] for r in json.loads(cache_path(set_name).read_text()) if part in r]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out


def floor_of(key: str) -> float:
    """The scale below which a difference does not count as relative (seconds, counts, shares)."""
    if "_s_p" in key or "_m_p" in key:
        return 1.0
    if "per_10min" in key or "per_live_minute" in key:
        return 0.5
    return 0.05


def relative(a: float | None, b: float | None, key: str) -> float | None:
    """|a - b| relative to the larger of the two (and a floor): 0 is equal, 1 is far apart."""
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), floor_of(key))


def similarity(target: dict, columns: dict[str, dict]) -> dict:
    """{aspect: {set: mean relative distance to the target}} over the aspect's statistics."""
    out = {}
    for aspect, keys in ASPECTS.items():
        row = {}
        for name, values in columns.items():
            distances = [relative(values.get(k), target.get(k), k) for k in keys]
            distances = [d for d in distances if d is not None]
            row[name] = float(np.mean(distances)) if distances else None
        out[aspect] = row
    return out


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if abs(value) < 100 else f"{value:.0f}"


def table(sets: list[str], truth: bool) -> None:
    others = [s for s in sets if s != "match" and cache_path(s).is_file()]
    target = set_means("match")
    v1_cols = {s: set_means(s) for s in others}
    truth_cols = {s: set_means(s, "truth") for s in others} if truth else {}
    names = ["match"] + [SHORT.get(s, s) for s in others]
    names += [f"{SHORT.get(s, s)}*" for s in truth_cols]
    print("statistic".ljust(30) + "".join(n.rjust(9) for n in names) + "   closest (V1 labels)")
    for aspect, keys in ASPECTS.items():
        print(f"-- {aspect}")
        for key in keys:
            values = [target.get(key)] + [v1_cols[s].get(key) for s in others]
            values += [truth_cols[s].get(key) for s in truth_cols]
            distances = {s: relative(v1_cols[s].get(key), target.get(key), key) for s in others}
            known = {s: d for s, d in distances.items() if d is not None}
            best = min(known, key=known.get) if known else "-"
            print(key.ljust(30) + "".join(_fmt(v).rjust(9) for v in values) + f"   {best}")
    if truth_cols:
        print("(* = the same statistic with the truth's labels instead of version 1's)")
        extra = sorted({k for c in truth_cols.values() for k in c if "@" in k})
        for key in extra:
            print(key.ljust(30) + "".join(_fmt(truth_cols[s].get(key)).rjust(9)
                                          for s in truth_cols))  # fmt: skip
    print("\nsimilarity to the match (mean relative distance per aspect, version 1's labels;"
          " 0 = the same):")  # fmt: skip
    sim = similarity(target, v1_cols)
    print("aspect".ljust(30) + "".join(SHORT.get(s, s).rjust(9) for s in others) + "   closest")
    for aspect, row in sim.items():
        known = {s: d for s, d in row.items() if d is not None}
        best = min(known, key=known.get) if known else "-"
        print(aspect.ljust(30) + "".join(_fmt(row[s]).rjust(9) for s in others) + f"   {best}")
    overall = {s: np.mean([row[s] for row in sim.values() if row[s] is not None]) for s in others}
    print("overall".ljust(30) + "".join(_fmt(overall[s]).rjust(9) for s in others))


# -- fitting the profile match_regime ------------------------------------------------------------

# What the fit matches (version 1's labels on both sides), by group; each group counts once in the
# loss, the agreement of our two pipelines twice (it is what decides how hard the task is).
FIT_GROUPS = {
    "ball lost in play": (
        "missing_given_live", "live_missing_free", "live_missing_lead_in", "live_gap_s_p50",
        "live_gap_s_p90",
    ),
    "ball lost in stoppages": (
        "missing_given_dead", "dead_tracked_first_third", "dead_tracked_middle_third",
        "dead_tracked_last_third", "stoppage_start_tracked", "stoppage_end_tracked",
    ),
    "ball in view in stoppages": (
        "dead_tracked_still", "dead_tracked_outside", "stoppage_end_still", "end_visible_s_p50",
    ),
    "tracked ball": (
        "still_given_tracked", "still_runs_per_10min", "reappear_speed_p50",
        "outside_given_tracked", "nearest_player_live_m_p50", "nearest_player_live_m_p90",
    ),
    "hardness": ("possession_agreement", "team_agreement", "team_margin"),
}  # fmt: skip
FIT_WEIGHTS = {"hardness": 2.0}
FIT_BASE = "label_free"  # the eight required keys not set below come from this profile
FIT_FIXED = {"free_gap_s": [4.0, 9.0], "placed_jitter_m": 0.05}  # from the match's live gaps
# Grid 1: how the ball is seen in stoppages, the rest held at a first guess (FIT_START).
FIT_START = {"lost_ball_share": 0.07, "lead_in_fraction": 0.3, "reacquire_s": [0.0, 1.0],
             "ball_offset_m": 1.5}  # fmt: skip
GRID_STOPPAGES = {
    "visible_dead": ((0.25, [0.5, 2.0]), (0.5, [0.5, 3.0]), (0.9, [1.0, 6.0])),
    "placed_ball_prob": (0.35, 0.65),
    "placed_ball_s": ([1.0, 4.0], [2.0, 8.0]),
    "carried_ball_s": ([0.0, 0.0], [2.0, 10.0]),
}
# Grid 2: how much of play loses the ball and how far it is off its possessor, the stoppages of
# the best point of grid 1 kept.
GRID_PLAY = {
    "lost_ball_share": (0.05, 0.08),
    "lead_in_fraction": (0.25, 0.5),
    "reacquire_s": ([0.0, 0.0], [0.0, 1.0]),
    "ball_offset_m": (1.0, 1.5, 2.0, 2.5),
}


def fit_loss(stats: dict, target: dict) -> float:
    """Weighted mean over FIT_GROUPS of the mean relative distance of the group's statistics."""
    total, weights = 0.0, 0.0
    for group, keys in FIT_GROUPS.items():
        distances = [relative(stats.get(k), target.get(k), k) for k in keys]
        distances = [d for d in distances if d is not None]
        if distances:
            weight = FIT_WEIGHTS.get(group, 1.0)
            total += weight * float(np.mean(distances))
            weights += weight
    return total / weights if weights else float("inf")


def group_distances(stats: dict, target: dict) -> dict:
    out = {}
    for group, keys in FIT_GROUPS.items():
        distances = [relative(stats.get(k), target.get(k), k) for k in keys]
        distances = [d for d in distances if d is not None]
        out[group] = round(float(np.mean(distances)), 3) if distances else None
    return out


def stoppage_points() -> list[dict]:
    return [
        {**FIT_FIXED, **FIT_START, "visible_dead_prob": prob, "visible_dead_s": seconds,
         "placed_ball_prob": placed, "placed_ball_s": still, "carried_ball_s": carried}
        for prob, seconds in GRID_STOPPAGES["visible_dead"]
        for placed in GRID_STOPPAGES["placed_ball_prob"]
        for still in GRID_STOPPAGES["placed_ball_s"]
        for carried in GRID_STOPPAGES["carried_ball_s"]
    ]  # fmt: skip


def play_points(best: dict) -> list[dict]:
    return [
        {**best, "lost_ball_share": share, "lead_in_fraction": fraction, "reacquire_s": late,
         "ball_offset_m": offset}
        for share in GRID_PLAY["lost_ball_share"]
        for fraction in GRID_PLAY["lead_in_fraction"]
        for late in GRID_PLAY["reacquire_s"]
        for offset in GRID_PLAY["ball_offset_m"]
    ]  # fmt: skip


def _run_grid(name: str, points: list[dict], target: dict, jobs: int) -> list[dict]:
    """calibrate_to_match on `points`, cached in OUT/fit_<name>.json (aggregates only)."""
    from possession import validation

    path = OUT / f"fit_{name}.json"
    if path.is_file():
        rows = json.loads(path.read_text())
        if [r["overrides"] for r in sorted(rows, key=lambda r: r["index"])] == points:
            return sorted(rows, key=lambda r: r["loss"])

    def progress(row):
        print(f"  {name} loss {row['loss']:.4f}  {group_distances(row, target)}", flush=True)

    rows = validation.calibrate_to_match(
        str(V1), str(OTHER), target, seeds=(0,), base=FIT_BASE, processes=jobs,
        progress=progress, points=points, vector=True, loss=fit_loss,
    )  # fmt: skip
    for row in rows:
        row["index"] = points.index(row["overrides"])
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def fit(jobs: int, write: bool) -> None:
    from possession import validation

    target = set_means("match")
    first = _run_grid("stoppages", stoppage_points(), target, jobs)
    print(f"grid 1 best: loss {first[0]['loss']}  {first[0]['overrides']}")
    second = _run_grid("play", play_points(first[0]["overrides"]), target, jobs)
    rows = sorted(first + second, key=lambda r: r["loss"])
    best = rows[0]
    print(f"best: loss {best['loss']}  {best['overrides']}")
    print("by group:", group_distances(best, target))
    keys = [k for group in FIT_GROUPS.values() for k in group]
    print("statistic".ljust(30) + "match".rjust(9) + "fitted".rjust(9))
    for key in keys:
        print(key.ljust(30) + _fmt(target.get(key)).rjust(9) + _fmt(best.get(key)).rjust(9))
    if not write:
        return
    params = {**validation.degradation_profile(FIT_BASE), **best["overrides"]}
    evidence = {
        "base": FIT_BASE,
        "configs": ["configs/hybrid_supervised.toml", "configs/unsup_v2.toml"],
        "games": list(validation.GAMES),
        "seeds": [0],
        "target_competition_match": {k: target.get(k) for k in keys},
        "fitted": {k: best.get(k) for k in keys},
        "loss": best["loss"],
        "loss_by_group": group_distances(best, target),
        "loss_definition": {"groups": {g: list(k) for g, k in FIT_GROUPS.items()},
                            "weights": FIT_WEIGHTS,
                            "distance": "|a - b| / max(|a|, |b|, floor); floor 1 for seconds and "
                                        "metres, 0.5 for counts, 0.05 for shares"},
        "grids": {"stoppages": {k: [list(v) if isinstance(v, tuple) else v for v in values]
                                for k, values in GRID_STOPPAGES.items()},
                  "play": {k: list(v) for k, v in GRID_PLAY.items()},
                  "fixed": FIT_FIXED, "start": FIT_START},
        "runners_up": [{"loss": r["loss"], "overrides": r["overrides"]} for r in rows[1:4]],
    }  # fmt: skip
    source = (
        "Fitted by `scripts/regime_compare.py fit` so that the ball regime of the degraded"
        " external matches, read through version 1's labels, matches the same statistics of the"
        " competition match (how often the ball is lost in play and in stoppages, where and how"
        " it is seen inside stoppages, placed balls, how it reappears, how far it is from the"
        " nearest player), together with the agreement of our two pipelines (as for"
        " `match_like`). Uses version 1's predicted labels of the match, no label of the match"
        " and no published score."
    )
    path = validation.write_profile("match_regime", params, source, evidence)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["measure", "table", "fit"])
    parser.add_argument("sets", nargs="*", default=list(SETS))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--truth", action="store_true", help="also the truth-label columns")
    parser.add_argument("--write", action="store_true", help="fit: write the profile")
    args = parser.parse_args()
    sets = args.sets or list(SETS)
    if args.action == "measure":
        measure(sets, args.jobs, args.force)
    elif args.action == "fit":
        fit(args.jobs, args.write)
    else:
        table(["match"] + [s for s in sets if s != "match"], args.truth)


if __name__ == "__main__":
    main()
