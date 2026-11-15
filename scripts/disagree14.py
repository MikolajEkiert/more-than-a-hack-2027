"""Where two configurations disagree about play and stoppage, and which looks right (task 14).

    uv run python scripts/disagree14.py CONFIG_A CONFIG_B [--set label_free|match_like|pff|idsse]

On the competition match (default) there are no labels, so the referee is label-free: the share
of frames with running players (90th percentile of the players' speed, 1 s mean, at least
2.5 m/s; `inspect_data._running`). On the labelled sets it reads 0.19-0.23 in true stoppages and
0.83-0.92 in true play. For every cell of the A-versus-B live/dead confusion it prints the share
of graded frames, the share with the ball tracked and the running share. With `--set`, the same
on the conditions of a labelled set, plus the share of each cell that is truly dead.

Aggregates only; nothing about single frames is printed.
"""

import argparse

import numpy as np

from possession import config as config_mod
from possession import inspect_data, pipeline, stages, validation  # noqa: F401  (registers)

SETS = {
    "label_free": {"suite": "metrica", "profile": "label_free"},
    "match_like": {"suite": "metrica", "profile": "match_like"},
    "pff": {"suite": "pff"},
    "idsse": {"suite": "idsse", "seeds": (0, 1)},
}


def cells(match, a, b, truth=None) -> dict:
    graded = match.graded
    live_a, live_b = np.isin(a, ("H", "A")) & graded, np.isin(b, ("H", "A")) & graded
    running = inspect_data._running(match)
    seen = match.ball_seen
    out = {}
    for name, mask in (
        ("both live", live_a & live_b),
        ("both dead", graded & ~live_a & ~live_b),
        ("A live, B dead", live_a & ~live_b),
        ("A dead, B live", ~live_a & live_b & graded),
    ):
        n = int(mask.sum())
        row = {
            "frames": n,
            "share": n / max(int(graded.sum()), 1),
            "ball_tracked": float(seen[mask].mean()) if n else None,
            "running": float(running[mask].mean()) if n else None,
        }
        if truth is not None:
            row["truly_dead"] = float((truth[mask] == "D").mean()) if n else None
        out[name] = row
    return out


def pooled(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        total = sum(r[key]["frames"] for r in rows)
        out[key] = {"frames": total}
        for stat in ("share", "ball_tracked", "running", "truly_dead"):
            if stat not in rows[0][key]:
                continue
            values = [(r[key][stat], r[key]["frames"]) for r in rows if r[key][stat] is not None]
            weight = sum(w for _, w in values)
            if stat == "share":
                out[key][stat] = float(np.mean([r[key]["share"] for r in rows]))
            else:
                out[key][stat] = sum(v * w for v, w in values) / weight if weight else None
    return out


def show(title: str, table: dict) -> None:
    print(title)
    for key, row in table.items():
        parts = [f"{key:<16} share {row['share']:.4f}"]
        for stat in ("ball_tracked", "running", "truly_dead"):
            if stat in row:
                value = row[stat]
                parts.append(f"{stat} {value:.3f}" if value is not None else f"{stat} -")
        print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_a")
    parser.add_argument("config_b")
    parser.add_argument("--set", default=None, choices=sorted(SETS))
    args = parser.parse_args()
    cfg_a, cfg_b = config_mod.load(args.config_a), config_mod.load(args.config_b)
    if args.set is None:
        from possession import loader

        match = loader.load_match(verbose=False)
        a = pipeline.run(cfg_a, match).labels
        b = pipeline.run(cfg_b, match).labels
        show(f"competition match, A = {cfg_a.name}, B = {cfg_b.name}", cells(match, a, b))
        return
    rows = []
    for _, _, match, truths in validation.conditions(**SETS[args.set]):
        truth = next(iter(truths.values()))  # stoppages are the same under every reading
        a = pipeline.run(cfg_a, match).labels
        b = pipeline.run(cfg_b, match).labels
        rows.append(cells(match, a, b, truth))
    show(f"{args.set}, A = {cfg_a.name}, B = {cfg_b.name} (pooled)", pooled(rows))


if __name__ == "__main__":
    main()
