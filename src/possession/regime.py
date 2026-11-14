"""The ball regime of automatic tracking, applied to clean training matches (task 12).

Corrected tracking (IDSSE, the supervised model's training data) has the ball on every frame,
stoppages included, and the package's degradation (`solution/degrade.py`) removes it without
looking at the labels. Automatic tracking loses the ball during most stoppages: P(ball missing |
dead) is 0.99 on the PFF broadcast tracking and 0.87-0.96 on the degraded Metrica matches, against
0.42 on IDSSE as degraded for training (docs/DECISIONS.md, task 11). A model trained on the latter
learns stoppages from where the ball is and how it moves, not from the ball disappearing.

`automatic_ball` removes the ball of a clean package container the way the degradation profiles of
the Metrica matches do (configs/degradation/, `validation.degrade`), but starting from a ball that
exists during stoppages:

  * during every true stoppage the ball is lost, except that after `visible_dead_prob` of the
    stoppages that follow play it stays in view for `visible_dead_s` (here the real ball is kept);
  * after the restart it is picked up again only after `REACQUIRE_S` (a ball reappears moving);
  * `lost_ball_share` of the live frames lose the ball: half just before stoppages (`lead_in_s`),
    half in stretches of `free_gap_s` anywhere in play.

Only the ball's x and y change; labels, players and everything else are untouched. The package's
own degradation then adds its position noise and player dropout with `ball_missing=0`
(`regime_params`), so that nothing is removed at random on top.

This module does not import the package: it works on its container layout (a dict of numpy arrays,
label codes H 0, A 1, D 2, X 3), and `possession.supervised` is the only bridge to it.
"""

import hashlib
import inspect
import json
import sys

import numpy as np

from possession import features, validation

REGIMES = {"auto_lf": "label_free", "auto_ml": "match_like"}  # regime -> degradation profile
MIXED = {"mix_lf": ("none", "auto_lf")}  # regime per training noise seed (seeds 0 and 1)
FPS = 25  # the package's containers are all at 25 frames per second
H, A, D = 0, 1, 2  # the package's label codes
REACQUIRE_S = (0.0, 0.5)


def seed_regime(regime: str, seed: int) -> str:
    """The ball regime that noise seed `seed` of training regime `regime` uses."""
    if regime in MIXED:
        return MIXED[regime][seed % len(MIXED[regime])]
    if regime == "none" or regime in REGIMES:
        return regime
    raise KeyError(f"unknown regime {regime!r}; known: none, {sorted(REGIMES)}, {sorted(MIXED)}")


def automatic_ball(m: dict, regime: str, seed: int) -> dict:
    """A copy of container `m` whose ball is lost as automatic tracking loses it (module doc)."""
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}; known: {sorted(REGIMES)}")
    p = validation.degradation_profile(REGIMES[regime])
    rng = np.random.default_rng(seed)
    label, half = np.asarray(m["label"]), np.asarray(m["half"])
    n = len(label)
    play = half > 0
    dead = (label == D) & play
    alive = np.isin(label, (H, A)) & play
    hidden = np.zeros(n, dtype=bool)
    starts, stops = features.runs(dead)
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        keep = 0
        if start > 0 and alive[start - 1] and rng.random() < p["visible_dead_prob"]:
            keep = int(rng.uniform(*p["visible_dead_s"]) * FPS)
        hidden[min(start + keep, stop) : stop] = True
        hidden[stop : min(stop + int(rng.uniform(*REACQUIRE_S) * FPS), n)] = True
    target = p["lost_ball_share"] * alive.sum()
    for index in rng.permutation(len(starts)).tolist():
        if (hidden & alive).sum() >= target / 2:
            break
        start = int(starts[index])
        hidden[max(start - int(rng.uniform(*p["lead_in_s"]) * FPS), 0) : start] = True
    alive_frames = np.flatnonzero(alive)
    for _ in range(10 * len(starts) + 100):
        if (hidden & alive).sum() >= target or not len(alive_frames):
            break
        start = int(rng.choice(alive_frames))
        hidden[start : start + int(rng.uniform(*p["free_gap_s"]) * FPS)] = True
    bx, by = np.array(m["bx"], copy=True), np.array(m["by"], copy=True)
    bx[hidden], by[hidden] = np.nan, np.nan
    return {**m, "bx": bx, "by": by}


def regime_key(regime: str) -> str:
    """Changes whenever this module, the regime's profile or the regime's definition changes."""
    if regime == "none":
        return "none"
    parts = [regime, inspect.getsource(sys.modules[__name__])]
    for name in sorted({seed_regime(regime, s) for s in (0, 1)} - {"none"}):
        parts.append(json.dumps(validation.degradation_profile(REGIMES[name]), sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def regime_inputs(containers: list, regime: str, seed: int) -> list:
    """The training containers of noise seed `seed` under `regime` (unchanged for "none")."""
    name = seed_regime(regime, seed)
    if name == "none":
        return containers
    return [automatic_ball(m, name, 1000 * seed + i) for i, m in enumerate(containers)]


def regime_params(params: dict, regime: str, seed: int) -> dict:
    """The package's degradation parameters for that seed: no random ball loss on top of the
    automatic one, and a tag that keeps the package's cache keys of the regimes apart."""
    name = seed_regime(regime, seed)
    if name == "none":
        return params
    return {**params, "ball_missing": 0.0, "regime": regime_key(name)}
