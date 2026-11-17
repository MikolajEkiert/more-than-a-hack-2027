"""Task 14: how much the decision protocol can see, from the cells already measured.

    uv run python scripts/protocol_check.py noise [MEMBER ...]
    uv run python scripts/protocol_check.py simulate [--draws N]
    uv run python scripts/protocol_check.py ranking [MEMBER ...] [--sets S,...]
    uv run python scripts/protocol_check.py rule-c MEMBER [MEMBER ...] --stage ball_state|team

Reads the per-condition cells of .cache/family/<member>/<set>.json (scripts/grid14.py,
scripts/family_grid.py) and prints aggregates only:

- `noise`: for each member against version 1 on the Metrica evidence, the paired difference and
  how much of it is noise: the spread between degradation seeds (same match, same reading), between
  matches, of the split between the two readings, and the standard error of a bootstrap over
  (match, 10-minute window) blocks; the same per-match spread on IDSSE and PFF.
- `simulate`: how often rule A (docs/FUSION.md section 10) and rule B (tasks/14) accept a change
  whose true effect is +0.00 to +0.04, under a noise model with the components pooled by `noise`.
- `ranking`: whether the sets order the members the same way (Kendall's tau of the members'
  mean differences to version 1 between two sets).
- `rule-c`: the verification proposed in docs/FUSION.md section 11 (a proposal, not a rule of the
  team): evidence from the profile fitted to the match's own ball regime, readings weighted
  towards `controlled`, thresholds set from the measured noise, vetoes only from the sets that
  resemble the match on what the change acts on.
"""

import argparse
import importlib.util
import itertools
import json

import numpy as np

from possession._paths import CACHE_DIR, REPO_ROOT

OUT = CACHE_DIR / "family"
EVIDENCE = ("label_free", "match_like")
READINGS = ("touch", "controlled")
FAMILY = ("M3", "M4", "M5", "M6")  # tasks 12-13, measured on every set
N_BOOT = 2000


def load(member: str, set_name: str) -> list[dict] | None:
    path = OUT / member / f"{set_name.replace(':', '@')}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def members() -> list[str]:
    return sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name != "V1")


def _by_key(cells):
    return {(c["game"], c["seed"], c["reading"]): c for c in cells}


def metrica_array(member: str, profile: str):
    """d[game, seed, reading] = member - V1 on one profile; None if not measured."""
    a, b = load(member, profile), load("V1", profile)
    if a is None or b is None:
        return None
    ka, kb = _by_key(a), _by_key(b)
    games = sorted({k[0] for k in kb})
    seeds = sorted({k[1] for k in kb})
    d = np.full((len(games), len(seeds), len(READINGS)), np.nan)
    for i, g in enumerate(games):
        for j, s in enumerate(seeds):
            for k, r in enumerate(READINGS):
                if (g, s, r) in ka and (g, s, r) in kb:
                    d[i, j, k] = ka[(g, s, r)]["spellF1"] - kb[(g, s, r)]["spellF1"]
    return d if np.isfinite(d).all() else None


def window_se(member: str, set_name: str) -> float | None:
    """Standard error of the mean difference from a bootstrap over (match, window) blocks."""
    a, b = load(member, set_name), load("V1", set_name)
    if a is None or b is None or "windows" not in a[0] or "windows" not in b[0]:
        return None
    ka, kb = _by_key(a), _by_key(b)
    blocks: dict = {}
    for key in sorted(set(ka) & set(kb)):
        for i, (x, y) in enumerate(zip(ka[key]["windows"], kb[key]["windows"], strict=True)):
            if x is not None and y is not None:
                blocks.setdefault((key[0], i), []).append(x - y)
    values = np.array([np.mean(v) for v in blocks.values()])
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return float(np.std(means))


def per_match(member: str, set_name: str) -> np.ndarray | None:
    """Mean difference to V1 per match (over seeds and the readings that count)."""
    from possession import validation

    a, b = load(member, set_name), load("V1", set_name)
    if a is None or b is None:
        return None
    suite = "metrica" if set_name in EVIDENCE or ":" in set_name else set_name
    suite = "idsse" if suite == "idsse_auto" else suite
    readings = validation.rule_readings(suite)
    ka, kb = _by_key(a), _by_key(b)
    games = sorted({k[0] for k in kb})
    out = []
    for g in games:
        diffs = [ka[k]["spellF1"] - kb[k]["spellF1"] for k in kb if k[0] == g and k in ka
                 and k[2] in readings]  # fmt: skip
        out.append(np.mean(diffs) if diffs else np.nan)
    return np.array(out)


# -- noise ---------------------------------------------------------------------------------------


def components(d: np.ndarray) -> dict:
    """Noise components of one member's d[game, seed, reading] on one profile."""
    over_readings = d.mean(axis=2)  # [game, seed]
    seed_sd = float(np.sqrt(np.mean(np.var(over_readings, axis=1, ddof=1))))
    per_game = over_readings.mean(axis=1)
    split = (d[:, :, 1] - d[:, :, 0]).mean(axis=1) / 2  # controlled minus touch, per game
    return {
        "mean": float(d.mean()),
        "by_reading": [float(d[:, :, k].mean()) for k in range(d.shape[2])],
        "per_game": [float(v) for v in per_game],
        "seed_sd": seed_sd,  # one seed's difference around the mean of the three (same match)
        "game_sd": float(np.std(per_game, ddof=1)),  # one match's difference around the mean
        "split_sd": float(np.std(split, ddof=1)),  # the reading split from match to match
        "split_mean": float(split.mean()),
    }


def pooled_noise(names: list[str]) -> dict:
    """Noise components pooled over members and both evidence profiles (root mean square)."""
    seeds, games, splits, profile_corr, ses = [], [], [], [], []
    for name in names:
        per_profile = {}
        for profile in EVIDENCE:
            d = metrica_array(name, profile)
            if d is None:
                continue
            c = components(d)
            seeds.append(c["seed_sd"])
            games.append(c["game_sd"] ** 2 - c["seed_sd"] ** 2 / d.shape[1])
            splits.append(c["split_sd"])
            per_profile[profile] = np.array(c["per_game"]) - c["mean"]
            se = window_se(name, profile)
            if se is not None:
                ses.append(se)
        if len(per_profile) == 2:
            a, b = per_profile.values()
            norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if norm > 0:  # a copy of V1 has no spread at all
                profile_corr.append(float(np.sum(a * b) / norm))
    return {
        "seed_sd": float(np.sqrt(np.mean(np.square(seeds)))),
        "game_sd": float(np.sqrt(max(np.mean(games), 0.0))),  # beyond the seed noise
        "split_sd": float(np.sqrt(np.mean(np.square(splits)))),
        "profile_corr": float(np.mean(profile_corr)) if profile_corr else 0.0,
        "window_se": float(np.sqrt(np.mean(np.square(ses)))) if ses else None,
        "n_members": len(names),
    }


def veto_sd(names: list[str], set_name: str) -> float | None:
    """Per-match spread of a member's difference to V1 around its mean (pooled)."""
    values = []
    for name in names:
        diffs = per_match(name, set_name)
        if diffs is not None and np.isfinite(diffs).all() and len(diffs) > 1:
            values.append(np.var(diffs, ddof=1))
    return float(np.sqrt(np.mean(values))) if values else None


def noise(names: list[str]) -> dict:
    print("member      profile     mean    touch   contr.  per match (3)              "
          "seed_sd  game_sd  split_sd  window_se")  # fmt: skip
    usable = []
    for name in names:
        ok = False
        for profile in EVIDENCE:
            d = metrica_array(name, profile)
            if d is None:
                continue
            ok = True
            c = components(d)
            se = window_se(name, profile)
            games = " ".join(f"{v:+.4f}" for v in c["per_game"])
            print(f"{name:<11} {profile:<11} {c['mean']:+.4f} {c['by_reading'][0]:+.4f} "
                  f"{c['by_reading'][1]:+.4f}  {games}   {c['seed_sd']:.4f}   {c['game_sd']:.4f}"
                  f"   {c['split_sd']:.4f}    {'-' if se is None else f'{se:.4f}'}")  # fmt: skip
        if ok:
            usable.append(name)
    pooled = pooled_noise(usable)
    pooled["pff_match_sd"] = veto_sd(usable, "pff")
    pooled["idsse_match_sd"] = veto_sd(usable, "idsse")
    pooled["idsse_auto_match_sd"] = veto_sd(usable, "idsse_auto")
    print("\npooled over", usable)
    for key, value in pooled.items():
        print(
            f"  {key:<20} {value if value is None or isinstance(value, int) else round(value, 4)}"
        )
    sd_mean = np.sqrt(pooled["game_sd"] ** 2 / 3 + pooled["seed_sd"] ** 2 / 9)
    print(f"  standard error of a profile's mean over 3 matches x 3 seeds: {sd_mean:.4f}")
    print(f"  of the mean over both profiles (correlation {pooled['profile_corr']:.2f}): "
          f"{sd_mean * np.sqrt((1 + pooled['profile_corr']) / 2):.4f}")  # fmt: skip
    return pooled


# -- simulation --------------------------------------------------------------------------------


def simulate(pooled: dict, effects=(0.0, 0.01, 0.02, 0.03, 0.04), draws: int = 20000,
             split: float = 0.0, idsse_effect: float | None = None) -> dict:  # fmt: skip
    """Acceptance rates of rule A and rule B for a change of true effect `effect` everywhere
    (on IDSSE `idsse_effect` if given), `split` added to controlled and taken from touch."""
    rng = np.random.default_rng(0)
    n_games, n_seeds = 3, 3
    out = {}
    for effect in effects:
        # match effects, correlated between the two profiles
        rho = max(min(pooled["profile_corr"], 0.999), 0.0)
        shared = rng.normal(size=(draws, 1, n_games))
        own = rng.normal(size=(draws, 2, n_games))
        game = pooled["game_sd"] * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * own)
        seed = rng.normal(0, pooled["seed_sd"], size=(draws, 2, n_games, n_seeds))
        reading = rng.normal(0, pooled["split_sd"], size=(draws, 2, n_games))
        base = effect + game[..., None] + seed  # [draw, profile, game, seed]
        touch = base - (split + reading)[..., None]
        controlled = base + (split + reading)[..., None]
        d = np.stack([touch, controlled], axis=-1)  # [draw, profile, game, seed, reading]
        by_profile = d.mean(axis=(2, 3, 4))
        by_profile_reading = d.mean(axis=(2, 3))
        by_game = d.mean(axis=(1, 3, 4))
        evidence_a = (
            (by_profile >= 0.02).all(axis=1)
            & (by_profile_reading >= 0).all(axis=(1, 2))
            & ((by_game > 0).sum(axis=1) >= 2)
        )
        pff = effect + rng.normal(0, pooled["pff_match_sd"], size=(draws, 16))
        e_idsse = effect if idsse_effect is None else idsse_effect
        idsse = e_idsse + rng.normal(0, pooled["idsse_match_sd"], size=(draws, 7))
        auto = effect + rng.normal(0, pooled["idsse_auto_match_sd"], size=(draws, 7))
        veto_pff = (pff.mean(axis=1) >= 0) & ((pff > 0).sum(axis=1) >= 9)
        veto_idsse = (idsse.mean(axis=1) >= 0) & ((idsse > 0).sum(axis=1) >= 5)
        mean_b = by_profile.mean(axis=1)
        bound = mean_b - 1.645 * pooled["window_se"] * np.sqrt((1 + rho) / 2)
        evidence_b = (
            (mean_b >= 0.010) & (bound > 0) & (by_profile_reading >= -0.010).all(axis=(1, 2))
        )
        vetoes_b = veto_pff & (auto.mean(axis=1) >= 0) & (idsse.mean(axis=1) >= -0.010)
        out[effect] = {
            "A evidence": float(evidence_a.mean()),
            "A with vetoes": float((evidence_a & veto_pff & veto_idsse).mean()),
            "B evidence": float(evidence_b.mean()),
            "B with vetoes": float((evidence_b & vetoes_b).mean()),
        }
    return out


def print_simulation(pooled: dict, draws: int) -> None:
    for title, kw in (
        ("same effect everywhere, no reading split", {}),
        ("reading split: touch 0.025 below the effect, controlled 0.025 above it",
         {"split": 0.025}),
        ("no effect on plain IDSSE (a ball-state change in the match's regime)",
         {"idsse_effect": 0.0}),
    ):  # fmt: skip
        print(f"\nacceptance rate, {title}:")
        result = simulate(pooled, draws=draws, **kw)
        print("true effect   " + "  ".join(f"{k:>14}" for k in next(iter(result.values()))))
        for effect, row in result.items():
            print(f"{effect:+.2f}         " + "  ".join(f"{v:>14.3f}" for v in row.values()))


# -- ranking -----------------------------------------------------------------------------------


def kendall(a: list[float], b: list[float]) -> float:
    pairs = list(itertools.combinations(range(len(a)), 2))
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in pairs)
    return float(s / len(pairs)) if pairs else float("nan")


def ranking(names: list[str], sets: list[str]) -> None:
    table = {}
    for name in ["V1", *names]:
        row = {}
        for set_name in sets:
            if name == "V1":
                row[set_name] = 0.0 if load("V1", set_name) else None
                continue
            diffs = per_match(name, set_name)
            row[set_name] = None if diffs is None else float(np.nanmean(diffs))
        if all(v is not None for v in row.values()):
            table[name] = row
    print("difference to V1 (mean over matches)")
    print("member".ljust(16) + "".join(s.rjust(14) for s in sets))
    for name, row in table.items():
        print(name.ljust(16) + "".join(f"{row[s]:+14.4f}" for s in sets))
    print(f"\nKendall's tau of the {len(table)} members' order between two sets (the share of"
          " member pairs ordered the same way is (1 + tau) / 2):")  # fmt: skip
    print("".ljust(16) + "".join(s.rjust(14) for s in sets))
    names_ = list(table)
    for a in sets:
        cells = []
        for b in sets:
            x = [table[n][a] for n in names_]
            y = [table[n][b] for n in names_]
            cells.append(f"{kendall(x, y):+.2f}")
        print(a.ljust(16) + "".join(c.rjust(14) for c in cells))


# -- rule C (proposal, docs/FUSION.md section 11) -----------------------------------------------

C_PRIMARY = "metrica:match_regime"
C_WEIGHTS = {"touch": 1 / 3, "controlled": 2 / 3}
C_MIN_GAIN = 0.010  # held-out weighted mean over the primary evidence, at least
C_SE = 0.004  # pooled standard error of one profile's mean (noise); the threshold grows with K
C_PRIMARY_FLOOR = -0.005  # no reading under the primary profile below this
C_REPORTED_FLOOR = -0.010  # no reading under the other Metrica profiles below this
C_GROSS_LOSS = -0.020  # any set losing more than this blocks (a bug, not a domain difference)
# The sets that resemble the match on what a change acts on (section 11, regime_compare.py table):
# (other evidence profiles, veto sets with the matches they must win, sets reported only).
C_STAGES = {
    "ball_state": (("label_free",), {"pff": 9}, ("match_like", "idsse", "idsse_auto")),
    "team": (("match_like",), {"idsse": 5}, ("label_free", "idsse_auto", "pff")),
}


def _weighted(cells: dict, game: str) -> float:
    """Reading-weighted mean over seeds of one match's cells {(game, seed, reading): cell}."""
    out = 0.0
    for reading, weight in C_WEIGHTS.items():
        values = [c["spellF1"] for k, c in cells.items() if k[0] == game and k[2] == reading]
        out += weight * float(np.mean(values))
    return out


def rule_c(names: list[str], stage: str, compared: int | None = None) -> None:
    others, vetoes, reported = C_STAGES[stage]
    ref = _by_key(load("V1", C_PRIMARY))
    cells = {name: _by_key(load(name, C_PRIMARY)) for name in names}
    games = sorted({k[0] for k in ref})

    def score(name, chosen):
        return float(np.mean([_weighted(cells[name], g) - _weighted(ref, g) for g in chosen]))

    held, blocks, by_reading = {}, [], {r: [] for r in C_WEIGHTS}
    for game in games:
        pick = max(names, key=lambda m: score(m, [g for g in games if g != game]))
        held[game] = (pick, score(pick, [game]))
        mine = {k: c for k, c in cells[pick].items() if k[0] == game}
        for reading in C_WEIGHTS:
            diffs = [mine[k]["spellF1"] - ref[k]["spellF1"] for k in mine if k[2] == reading]
            by_reading[reading].append(float(np.mean(diffs)))
        windows: dict = {}
        for key, cell in mine.items():
            for i, (x, y) in enumerate(zip(cell["windows"], ref[key]["windows"], strict=True)):
                if x is not None and y is not None:
                    windows.setdefault(i, []).append(C_WEIGHTS[key[2]] * 2 * (x - y))
        blocks += [float(np.mean(v)) for v in windows.values()]
    gain = float(np.mean([v for _, v in held.values()]))
    values = np.array(blocks)
    rng = np.random.default_rng(0)
    low = float(np.percentile(values[rng.integers(0, len(values), (N_BOOT, len(values)))]
                              .mean(axis=1), 5))  # fmt: skip
    readings = {r: float(np.mean(v)) for r, v in by_reading.items()}
    print("held-out picks:", {g: p for g, (p, _) in held.items()})
    print(f"primary ({C_PRIMARY}, readings {C_WEIGHTS}): held-out gain {gain:+.4f}, "
          f"5% bound {low:+.4f}, by reading " + ", ".join(f"{r} {v:+.4f}"
                                                        for r, v in readings.items()))  # fmt: skip
    k = max(compared or len(names), 1)
    need = max(C_MIN_GAIN, C_SE * float(np.sqrt(2 * np.log(k))))
    print(f"threshold for {k} configurations compared: {need:+.4f}")
    ok = gain >= need and low > 0 and all(v >= C_PRIMARY_FLOOR for v in readings.values())
    final = max(names, key=lambda m: score(m, games))
    print(f"evidence passes: {ok}; member chosen on all three matches: {final}")
    for profile in others:
        a, b = _by_key(load(final, profile)), _by_key(load("V1", profile))
        diffs = {r: float(np.mean([a[k]["spellF1"] - b[k]["spellF1"] for k in b if k[2] == r]))
                 for r in C_WEIGHTS}  # fmt: skip
        passes = all(v >= C_REPORTED_FLOOR for v in diffs.values())
        ok = ok and passes
        print(f"{profile}: " + ", ".join(f"{r} {v:+.4f}" for r, v in diffs.items())
              + f" -> {'passes' if passes else 'FAIL'}")  # fmt: skip
    for set_name, need in vetoes.items():
        diffs = per_match(final, set_name)
        passes = bool(np.mean(diffs) >= 0 and (diffs > 0).sum() >= need)
        ok = ok and passes
        print(f"veto {set_name}: {np.mean(diffs):+.4f}, won {(diffs > 0).sum()}/{len(diffs)} -> "
              f"{'passes' if passes else 'VETO'}")  # fmt: skip
    for set_name in reported:
        diffs = per_match(final, set_name)
        if diffs is None:
            continue
        passes = bool(np.mean(diffs) >= C_GROSS_LOSS)
        ok = ok and passes
        print(f"reported {set_name}: {np.mean(diffs):+.4f}, won {(diffs > 0).sum()}/{len(diffs)}"
              + ("" if passes else " -> gross loss, BLOCKS"))  # fmt: skip
    spec = importlib.util.spec_from_file_location("grid14", REPO_ROOT / "scripts" / "grid14.py")
    grid14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid14)
    fp = grid14.footprint_of(grid14.config_of(final))
    inside = all(lo <= fp[k] <= hi for k, (lo, hi) in grid14.FOOTPRINT.items())
    ok = ok and inside
    print("footprint:", {k: fp[k] for k in grid14.FOOTPRINT}, "->", "inside" if inside else "OUT")
    print("VERDICT (rule C, proposal):", f"change to {final}" if ok else "version 1 stays")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["noise", "simulate", "ranking", "rule-c"])
    parser.add_argument("members", nargs="*")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--sets", default="label_free,match_like,idsse,idsse_auto,pff")
    parser.add_argument("--stage", default="ball_state", choices=sorted(C_STAGES))
    parser.add_argument("--compared", type=int, help="rule-c: configurations compared (K)")
    args = parser.parse_args()
    names = args.members or members()
    if args.action == "noise":
        noise(names)
    elif args.action == "simulate":
        print_simulation(noise(names), args.draws)
    elif args.action == "rule-c":
        rule_c(names, args.stage, args.compared)
    else:
        ranking(names, [s for s in args.sets.split(",") if s])


if __name__ == "__main__":
    main()
