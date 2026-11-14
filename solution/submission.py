"""Train on all of IDSSE, predict BAR-ATH, decode, write the 199,319-row submission.

    python -m solution.submission --team MyTeam                   # v1 features, defaults
    python -m solution.submission --team MyTeam --features v0 --pair 4 32

The notebook calls the same functions; this CLI is what the repository's reproduction
instructions point at (RULES.md: running the repository must regenerate the identical file), so
every choice is an argument with a default, and everything random is seeded.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import (atd, cache, container as C, decode, degrade, evaluate_external as ev,  # noqa: E402
                      features, features_seq, idsse, labels)  # features_seq registers "v2"

pm = C.pm

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
MODEL_DIR = os.path.join(ROOT, "data", "models")
DEFAULT_PAIR = (4, 32)              # harness best on v0; replace with the notebook's choice
DEGRADE_SEEDS = (0, 1)              # two noise realisations of every training match
TEAM_TAG = {"barcelona": C.H, "atletic club": C.A, "athletic club": C.A}
ATTACKING = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


PFF_STRIDE = 15          # 64 PFF matches: every 15th frame, one orientation (RAM)


def train_final(feature_fn=features.build_features_v1, label_fn=labels.raw, degrade_params=None,
                seeds=DEGRADE_SEEDS, log=print, extra=None, extra_stride=PFF_STRIDE, train_cfg=None):
    """One model on all seven IDSSE matches (both orientations, several noise draws), plus
    `extra` match loaders (e.g. PFF, one orientation, one draw). Cached."""
    extra = extra or []
    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))

    def fit():
        Xs, ys, ws = [], [], []
        for seed in seeds:
            data = ev.prepare(idsse.load_all(), feature_fn, label_fn, seed=seed,
                              degrade_params=degrade_params, log=log)
            X, y = ev.stack(data)
            ws.append(ev.stack_weights(data, cfg=cfg))
            del data
            Xs.append(X)
            ys.append(y)
        if extra:
            more = ev.prepare(extra, feature_fn, label_fn, seed=seeds[0],
                              degrade_params=degrade_params, log=log, swap=False, light=True)
            X, y = ev.stack(more, stride=extra_stride)
            ws.append(ev.stack_weights(more, stride=extra_stride, cfg=cfg))
            del more
            Xs.append(X)
            ys.append(y)
        w = np.concatenate(ws) if cfg["weights"] else None
        return ev.train(np.concatenate(Xs), np.concatenate(ys), weight=w)

    full = sorted(dict(degrade.DEFAULTS, **(degrade_params or {})).items())
    parts = ["final", features.feature_key(feature_fn), labels.label_key(label_fn),
             full, list(seeds), cache.source(degrade), sorted(idsse.MATCHES),
             ev.LGB_PARAMS, ev.ROUNDS, ev.TRAIN_STRIDE]
    if extra:  # keep the IDSSE-only key unchanged so existing cached models stay valid
        ends = [f() if callable(f) else f for f in (extra[0], extra[-1])]
        parts += [len(extra), extra_stride, [cache.fingerprint(m) for m in ends]]
    if cfg != ev.DEFAULT_CFG:
        parts += [sorted(cfg.items())]
    return cache.model(cache.digest(*parts), fit)


def predict_match(model, feature_fn, m):
    """(T, 3) probabilities for a container; its features go through the cache too."""
    X, _ = features.cached(feature_fn, m, "orig", cache.digest("match", cache.fingerprint(m)))
    return model.predict(X)


def smart_tagging_prior(T, beta=0.1, game="BAR-ATH"):
    """(T, 3) additive log-prior from Smart Tagging phases. Zero where nothing is tagged.

    Attacking phases favour the tagged team, defending phases the other team. ASSUMPTION to check
    by watching a few: 'Team' on a defending phase is the defending side. Set pieces are left out:
    their windows open ~3.5 s before the restart, so they carry no clean team-or-dead signal.
    """
    import load_smart_tagging  # starter/
    prior = np.zeros((T, 3))
    for tag in load_smart_tagging.load(game):
        team = TEAM_TAG.get(str(tag["tags"].get("Team", "")).strip().lower())
        code = tag["code"].strip().lower()
        if team is None or code not in ATTACKING | DEFENDING:
            continue
        side = team if code in ATTACKING else (C.A if team == C.H else C.H)
        a, b = max(tag["start_frame"], 0), min(tag["end_frame"], T)
        prior[a:b, side] += beta
    return prior


def apply_prior(prob, prior):
    logp = np.log(np.clip(prob, 1e-6, 1)) + prior
    p = np.exp(logp - logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def to_codes(prob, half, pair):
    """Decode inside the halves; X elsewhere (never graded)."""
    return decode.decode(prob, half, [pair])[0]


def kickoff_rule(codes, half, game="BAR-ATH"):
    """D from the start of the graded second half up to its kickoff.

    FAQ.md: the second half is graded from its first frame but kicks off later (113,742), and
    every frame in between is D. The frame is the organisers' own published constant
    (starter/load_smart_tagging.py), not something read off the video or the ground truth.
    """
    import load_smart_tagging  # starter/: the organisers' published kickoff frames
    kick = load_smart_tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
    h2 = np.flatnonzero(half == 2)
    out = codes.copy()
    if len(h2) and h2[0] < kick:
        out[h2[0]:kick] = C.D
    return out


def write_csv(codes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = np.array(pm.LABELS)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, names[c]) for i, c in enumerate(codes))
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def summary(codes):
    runs = [(a, b) for h, a, b in __import__("games").half_video_blocks("BAR-ATH")]
    runs = [(a, b + 1) for a, b in runs]
    spells = pm.possession_spells(codes, runs)
    dur = np.array([(b - a) / C.FPS for a, b, _ in spells])
    graded = np.concatenate([codes[a:b] for a, b in runs])
    return {"possessions": len(spells), "mean_s": float(dur.mean()), "median_s": float(np.median(dur)),
            "share_H": float(np.mean(graded == C.H)), "share_A": float(np.mean(graded == C.A)),
            "share_D": float(np.mean(graded == C.D))}


# -- the final configuration -----------------------------------------------------------------
# Every default below was chosen on IDSSE leave-one-match-out (solution/experiments*.py); the
# public BAR-ATH window was never used to choose. IDSSE spellF1: 0.619 (v1) -> 0.626 (+weights)
# -> 0.637 (+boundary decoder, better on 7/7 held-out matches).
FINAL = {
    "features": "v1",              # v2 sequence features: -0.002, 3/7 -> rejected
    "weights": True,               # hand-over / short-possession sample weights: +0.007, 5/7
    "boundary": (8.0, 2.0, 32.0),  # P(hand-over)-aware switch costs (lambda, beta, dead): +0.012, 7/7
    "kickoff_rule": True,          # FAQ.md: graded second-half frames before kickoff are all D
    "degrade_json": os.path.join(ROOT, "data", "models", "degrade_params.json"),
    "with_pff": False,             # flip once PFF is downloaded AND the dataset matrix says it helps
}


def build(team, features_name=FINAL["features"], weights=FINAL["weights"],
          boundary_cfg=FINAL["boundary"], kickoff=FINAL["kickoff_rule"],
          degrade_json=FINAL["degrade_json"], with_pff=FINAL["with_pff"], pair=DEFAULT_PAIR,
          tau=0.0, smart_tagging=0.0, stack=False, log=print):
    """Train (cached), predict BAR-ATH, decode, write evaluation/results/<team>.csv.

    The single code path behind the CLI and the notebooks, so all of them produce the same file.
    Returns (codes, sha256).
    """
    if not atd.have_data():
        raise IOError("BAR-ATH data not found in ./data (see DATA.md)")
    params = json.load(open(degrade_json)) if degrade_json and os.path.isfile(degrade_json) else None
    feature_fn = features.FEATURE_SETS[features_name]
    label_fn = labels.harmonised(tau) if tau else labels.raw

    extra = None
    if with_pff:
        from solution import pff
        pff.convert_all(log=log)
        extra = pff.loaders()
        if not extra:
            raise IOError("with_pff: no PFF matches found under data/external/pff")
    model = train_final(feature_fn, label_fn, params, extra=extra, log=log,
                        train_cfg={"weights": True} if weights else None)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "{}.txt".format(team)))

    bar = atd.load()
    prob = predict_match(model, feature_fn, bar)
    if stack:
        from solution import stacking
        res = ev.run(feature_fn=feature_fn, label_fn=label_fn, degrade_params=params,
                     bootstrap=0, log=lambda s: None)          # cached out-of-fold probabilities
        stage2 = stacking.train_stage2(res["probs"], [d["m"]["label"] for d in res["data"]],
                                       [d["m"]["half"] for d in res["data"]])
        prob = stacking.apply(stage2, prob, bar["half"])
    if smart_tagging:
        prob = apply_prior(prob, smart_tagging_prior(len(prob), smart_tagging))
    if boundary_cfg:
        from solution import boundary
        bmodel = boundary.train_final(feature_fn, label_fn, params, extra=extra)
        X, _ = features.cached(feature_fn, bar, "orig", cache.digest("match", cache.fingerprint(bar)))
        codes = boundary.decode_with(prob, bar["half"], bmodel.predict(X), [tuple(boundary_cfg)])[0]
    else:
        codes = to_codes(prob, bar["half"], tuple(pair))
    if kickoff:
        codes = kickoff_rule(codes, bar["half"])
    digest = write_csv(codes, os.path.join(RESULTS_DIR, "{}.csv".format(team)))
    return codes, digest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Defaults are the final configuration: `--team NAME` alone builds the submission.")
    parser.add_argument("--team", required=True, help="file is written as <team>.csv")
    parser.add_argument("--features", default=FINAL["features"], choices=sorted(features.FEATURE_SETS))
    parser.add_argument("--no-weights", action="store_true", help="drop the hand-over sample weights")
    parser.add_argument("--boundary", type=float, nargs=3, default=FINAL["boundary"],
                        metavar=("LAMBDA", "BETA", "DEAD"), help="boundary-aware decoder settings")
    parser.add_argument("--plain-decoder", type=float, nargs=2, default=None, metavar=("TEAM", "DEAD"),
                        help="plain Viterbi with these penalties instead of the boundary decoder")
    parser.add_argument("--no-kickoff-rule", action="store_true",
                        help="do not force D before the second-half kickoff (FAQ.md)")
    parser.add_argument("--degrade-json", default=FINAL["degrade_json"],
                        help="calibrated degradation parameters (notebook section 3 writes them)")
    parser.add_argument("--with-pff", action="store_true",
                        help="also train on the PFF World Cup 2022 matches in data/external/pff")
    parser.add_argument("--tau", type=float, default=0.0, help="(rejected) label harmonisation, s")
    parser.add_argument("--smart-tagging", type=float, default=0.0, help="(off) prior weight")
    parser.add_argument("--stack", action="store_true", help="(rejected) stage-2 stacked model")
    args = parser.parse_args()
    try:
        codes, digest = build(
            args.team, features_name=args.features, weights=not args.no_weights,
            boundary_cfg=None if args.plain_decoder else args.boundary,
            pair=tuple(args.plain_decoder) if args.plain_decoder else DEFAULT_PAIR,
            kickoff=not args.no_kickoff_rule, degrade_json=args.degrade_json,
            with_pff=args.with_pff, tau=args.tau, smart_tagging=args.smart_tagging, stack=args.stack)
    except IOError as error:
        sys.exit(str(error))
    print(summary(codes))
    print("sha256", digest)


if __name__ == "__main__":
    main()
