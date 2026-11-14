"""The family of docs/FUSION.md section 10, measured and decided by its rule (tasks 12 and 13).

    uv run python scripts/family_grid.py measure MEMBER [MEMBER ...] [--sets S,...]
    uv run python scripts/family_grid.py decide MEMBER [MEMBER ...]

`measure` scores each member (and version 1, the reference "V1") on every condition of the sets
and keeps the cells in .cache/family/<member>/<set>.json, so that members added later (the
temporal network of task 13) are scored on exactly the same conditions. `decide` applies the rule:

1. On the Metrica matches (profiles label_free and match_like, readings touch and controlled) the
   member is chosen on two matches and scored on the third; the decision uses these held-out
   differences to version 1. Each profile must gain at least +0.02, no (profile, reading) may lose,
   and at least two of the three held-out matches must be won.
2. The member chosen on all three matches must not lose on the IDSSE matches out of fold (mean
   difference >= 0, 5 of 7 matches) nor on the PFF test matches (>= 0, 9 of 16).
3. Its footprint on the competition match: 3.5 to 7 possessions per live minute, dead share 0.35 to
   0.50.

Aggregates only; nothing is logged in experiments/runs/.
"""

import argparse
import json

import numpy as np

from possession import config as config_mod
from possession import pipeline, scoring, stages, validation  # noqa: F401  (registers methods)
from possession._paths import CACHE_DIR, REPO_ROOT

MEMBERS = {
    "V1": "configs/hybrid_supervised.toml",
    "M3": "configs/hybrid_regime_mix.toml",
    "M4": "configs/hybrid_tta.toml",
    "M5": "configs/hybrid_sequence_blend.toml",
    "M6": "configs/hybrid_sequence.toml",
}
SETS = {
    "label_free": {"suite": "metrica", "profile": "label_free"},
    "match_like": {"suite": "metrica", "profile": "match_like"},
    "idsse": {"suite": "idsse"},
    "idsse_auto": {"suite": "idsse_auto"},
    "pff": {"suite": "pff"},
}
EVIDENCE = ("label_free", "match_like")
VETOES = {"idsse": 5, "pff": 9}
MIN_GAIN, MIN_GAMES = 0.02, 2
FOOTPRINT = {"possessions_per_live_minute": (3.5, 7.0), "dead_share": (0.35, 0.50)}
OUT = CACHE_DIR / "family"


def measure(member: str, set_name: str) -> list[dict]:
    cfg = config_mod.load(REPO_ROOT / MEMBERS[member])
    kw = SETS[set_name]
    readings = validation.rule_readings(kw["suite"])
    cells = []
    for game, seed, match, truths in validation.conditions(**kw):
        labels = pipeline.run(cfg, match).labels
        for reading in readings:
            s = validation.score(labels, truths[reading])
            cells.append({"game": str(game), "seed": seed, "reading": reading, **s})
    path = OUT / member / f"{set_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cells))
    return cells


def load(member: str, set_name: str) -> list[dict]:
    return json.loads((OUT / member / f"{set_name}.json").read_text())


def mean(cells, **where) -> float:
    vals = [c["spellF1"] for c in cells if all(c[k] == v for k, v in where.items())]
    return float(np.mean(vals))


def evidence_diff(member, game, profile, reading) -> float:
    return mean(load(member, profile), game=game, reading=reading) - mean(
        load("V1", profile), game=game, reading=reading
    )


def decide(members: list[str]) -> None:
    games = sorted({c["game"] for c in load("V1", "label_free")})
    readings = validation.rule_readings("metrica")

    def score_on(member, chosen_games):
        return np.mean(
            [mean(load(member, p), game=g) for p in EVIDENCE for g in chosen_games]
        )  # fmt: skip

    print("in-sample differences to V1 (mean over the three matches):")
    for m in members:
        cells = [
            np.mean([evidence_diff(m, g, p, r) for g in games]) for p in EVIDENCE for r in readings
        ]
        print(f"  {m}: " + "  ".join(f"{p[:5]}/{r[:5]} {d:+.4f}" for (p, r), d in
              zip([(p, r) for p in EVIDENCE for r in readings], cells, strict=True)))  # fmt: skip
    held = {}
    for g in games:
        others = [x for x in games if x != g]
        pick = max(members, key=lambda m: score_on(m, others))
        held[g] = (pick, {(p, r): evidence_diff(pick, g, p, r) for p in EVIDENCE for r in readings})
        print(f"held out {g}: chosen {pick}")
    gain = {p: np.mean([d[(p, r)] for _, d in held.values() for r in readings]) for p in EVIDENCE}
    by = {
        (p, r): np.mean([d[(p, r)] for _, d in held.values()]) for p in EVIDENCE for r in readings
    }
    won = sum(np.mean(list(d.values())) > 0 for _, d in held.values())
    print("held-out gain by profile:", {p: round(float(v), 4) for p, v in gain.items()})
    print(
        "held-out gain by (profile, reading):",
        {f"{p}/{r}": round(float(v), 4) for (p, r), v in by.items()},
    )
    print("held-out matches won:", won, "of", len(games))
    ok = (
        all(v >= MIN_GAIN for v in gain.values())
        and all(v >= 0 for v in by.values())
        and won >= MIN_GAMES
    )
    final = max(members, key=lambda m: score_on(m, games))
    print(f"evidence passes: {ok}; member chosen on all three matches: {final}")
    for suite, need in VETOES.items():
        a, b = load(final, suite), load("V1", suite)
        names = sorted({c["game"] for c in b})
        diffs = [mean(a, game=n) - mean(b, game=n) for n in names]
        passes = np.mean(diffs) >= 0 and sum(d > 0 for d in diffs) >= need
        ok = ok and passes
        won_veto = sum(d > 0 for d in diffs)
        verdict = "passes" if passes else "VETO"
        print(f"veto {suite}: mean {np.mean(diffs):+.4f}, won {won_veto}/{len(names)} -> {verdict}")
    for suite in ("idsse_auto",):
        try:
            a, b = load(final, suite), load("V1", suite)
            diff = np.mean([c["spellF1"] for c in a]) - np.mean([c["spellF1"] for c in b])
            print(f"reported {suite}: {diff:+.4f}")
        except FileNotFoundError:
            pass
    from possession import loader

    match = loader.load_match(verbose=False)
    labels = pipeline.run(config_mod.load(REPO_ROOT / MEMBERS[final]), match).labels
    fp = scoring.footprint(labels, match.half, match.fps)
    inside = all(lo <= fp[k] <= hi for k, (lo, hi) in FOOTPRINT.items())
    ok = ok and inside
    print("footprint:", {k: fp[k] for k in FOOTPRINT}, "->", "inside" if inside else "OUTSIDE")
    print("VERDICT:", f"change to {final}" if ok else "version 1 stays")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["measure", "decide"])
    parser.add_argument("members", nargs="+")
    parser.add_argument("--sets", default=",".join(SETS))
    args = parser.parse_args()
    if args.action == "measure":
        for member in ["V1"] + [m for m in args.members if m != "V1"]:
            for s in args.sets.split(","):
                if member == "V1" and (OUT / member / f"{s}.json").exists():
                    continue
                cells = measure(member, s)
                print(f"{member} {s}: {np.mean([c['spellF1'] for c in cells]):.4f}", flush=True)
    else:
        decide(args.members)


if __name__ == "__main__":
    main()
