"""How much does each kind of mistake cost on spellF1?

Synthetic matches only. The generator's parameters come from published statistics about *other*
matches (see research/RESEARCH.md, section "Domain priors"), never from the public ground truth,
so nothing here is tuned on the provided game.

Each experiment takes a perfect synthetic labelling, damages it in exactly one way, and scores the
damage with the official metric library. Averaged over several random matches.

    python3 research/metric_sensitivity.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

FPS = 25
HALF_FRAMES = 79400
GAP = 20000
N_SEEDS = 6

# Siegle & Lames (2012): mean running segment 32.1 s, mean interruption 18.7 s, ~108 per match.
ALIVE_MEAN_S = 32.1
DEAD_MEAN_S = 18.7
# Opta sequence lengths: ~9 s league mean, 7-16 s team range. A lognormal with a heavy short tail,
# because deflections and failed controls produce many sub-2 s spells.
POSS_MEDIAN_S = 6.0
POSS_SIGMA = 1.0


def synthetic_match(rng):
    """Full-length label array (codes) with two graded halves and X elsewhere."""
    n = 2 * HALF_FRAMES + 3 * GAP
    labels = np.full(n, pm.X, dtype=np.int8)
    for half in range(2):
        start = GAP + half * (HALF_FRAMES + GAP)
        t = start
        stop = start + HALF_FRAMES
        team = rng.randint(2)
        while t < stop:
            alive = int(rng.gamma(2.0, ALIVE_MEAN_S / 2.0) * FPS) + FPS
            end_alive = min(stop, t + alive)
            while t < end_alive:
                d = int(rng.lognormal(np.log(POSS_MEDIAN_S), POSS_SIGMA) * FPS) + 5
                labels[t:min(end_alive, t + d)] = team
                t += d
                team = 1 - team
            t = end_alive
            if t >= stop:
                break
            dead = int(rng.gamma(2.0, DEAD_MEAN_S / 2.0) * FPS) + 2 * FPS
            labels[t:min(stop, t + dead)] = pm.D
            t += dead
            team = rng.randint(2)
    return labels


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def segs(codes, runs):
    return pm.segments(codes, runs)


def jitter_boundaries(gt, rng, sigma):
    """Move every label change by N(0, sigma) frames."""
    pred = gt.copy()
    for start, stop, code in segs(gt, pm.valid_runs(gt)):
        if start == 0:
            continue
        shift = int(round(rng.normal(0, sigma)))
        if shift > 0:
            pred[start:min(stop, start + shift)] = gt[start - 1]
        elif shift < 0:
            pred[max(0, start + shift):start] = code
    return pred


def constant_lag(gt, lag):
    pred = gt.copy()
    for start, stop in pm.valid_runs(gt):
        pred[start + lag:stop] = gt[start:stop - lag]
    return pred


def drop_short_spells(gt, min_s):
    """The model never sees possessions shorter than min_s: the previous team carries on."""
    pred = gt.copy()
    runs = pm.valid_runs(gt)
    for start, stop, code in segs(gt, runs):
        if code in (pm.H, pm.A) and stop - start < min_s * FPS and start > 0 \
                and pred[start - 1] in (pm.H, pm.A):
            pred[start:stop] = pred[start - 1]
    return pred


def insert_flicker(gt, rng, per_minute, dur_s=1.0):
    """Spurious opposite-team blips inside real possessions."""
    pred = gt.copy()
    alive = np.flatnonzero((gt == pm.H) | (gt == pm.A))
    n = int(len(alive) / (60 * FPS) * per_minute)
    width = int(dur_s * FPS)
    for f in rng.choice(alive, n, replace=False):
        if gt[f] in (pm.H, pm.A):
            seg = slice(f, f + width)
            pred[seg] = np.where(gt[seg] == pm.H, pm.A, np.where(gt[seg] == pm.A, pm.H, gt[seg]))
    return pred


def miss_dead_balls(gt, max_s):
    """Dead stretches shorter than max_s are predicted as the previous team still in possession."""
    pred = gt.copy()
    for start, stop, code in segs(gt, pm.valid_runs(gt)):
        if code == pm.D and stop - start < max_s * FPS and start > 0 and pred[start - 1] in (pm.H, pm.A):
            pred[start:stop] = pred[start - 1]
    return pred


def dead_as_abstain(gt):
    """Never say D: write X on every dead frame, so the scorer fills it with the last team."""
    pred = gt.copy()
    pred[gt == pm.D] = pm.X
    return pred


def shrink_dead(gt, s):
    """Dead stretches detected s seconds late at the start and s seconds early at the end."""
    pred = gt.copy()
    k = int(s * FPS)
    for start, stop, code in segs(gt, pm.valid_runs(gt)):
        if code == pm.D and start > 0 and stop - start > 2 * k:
            pred[start:start + k] = pred[start - 1]
            if stop < len(gt) and gt[stop] in (pm.H, pm.A):
                pred[stop - k:stop] = gt[stop]
    return pred


def wrong_restart_team(gt, rng, frac):
    """After a stoppage, give the first possession to the wrong team for a fraction of restarts."""
    pred = gt.copy()
    runs = pm.valid_runs(gt)
    s = segs(gt, runs)
    for i in range(1, len(s)):
        start, stop, code = s[i]
        if code in (pm.H, pm.A) and s[i - 1][2] == pm.D and rng.rand() < frac:
            pred[start:stop] = 1 - code
    return pred


def majority_smooth(gt, half_width):
    """What the baseline's majority vote does to an otherwise perfect signal."""
    out = gt.copy()
    for start, stop in pm.valid_runs(gt):
        w = gt[start:stop]
        cs = [np.concatenate(([0], np.cumsum(w == k))) for k in (pm.H, pm.A, pm.D)]
        idx = np.arange(len(w))
        lo = np.maximum(0, idx - half_width)
        hi = np.minimum(len(w), idx + half_width + 1)
        counts = np.stack([c[hi] - c[lo] for c in cs])
        out[start:stop] = np.array([pm.H, pm.A, pm.D])[np.argmax(counts, axis=0)]
    return out


def experiments():
    return [
        ("perfect", lambda g, r: g),
        ("boundary jitter sd 0.2 s", lambda g, r: jitter_boundaries(g, r, 5)),
        ("boundary jitter sd 0.5 s", lambda g, r: jitter_boundaries(g, r, 12)),
        ("boundary jitter sd 1 s", lambda g, r: jitter_boundaries(g, r, 25)),
        ("boundary jitter sd 2 s", lambda g, r: jitter_boundaries(g, r, 50)),
        ("constant lag 0.5 s", lambda g, r: constant_lag(g, 12)),
        ("constant lag 1 s", lambda g, r: constant_lag(g, 25)),
        ("miss possessions < 1 s", lambda g, r: drop_short_spells(g, 1)),
        ("miss possessions < 2 s", lambda g, r: drop_short_spells(g, 2)),
        ("miss possessions < 4 s", lambda g, r: drop_short_spells(g, 4)),
        ("flicker 0.5 / alive-min (1 s)", lambda g, r: insert_flicker(g, r, 0.5)),
        ("flicker 1 / alive-min (1 s)", lambda g, r: insert_flicker(g, r, 1.0)),
        ("flicker 3 / alive-min (1 s)", lambda g, r: insert_flicker(g, r, 3.0)),
        ("miss dead balls < 5 s", lambda g, r: miss_dead_balls(g, 5)),
        ("miss dead balls < 10 s", lambda g, r: miss_dead_balls(g, 10)),
        ("never say D (X on dead)", lambda g, r: dead_as_abstain(g)),
        ("dead stretch 1 s short each end", lambda g, r: shrink_dead(g, 1)),
        ("dead stretch 3 s short each end", lambda g, r: shrink_dead(g, 3)),
        ("wrong team on 10% of restarts", lambda g, r: wrong_restart_team(g, r, 0.1)),
        ("wrong team on 30% of restarts", lambda g, r: wrong_restart_team(g, r, 0.3)),
        ("majority vote +/-1 s on perfect", lambda g, r: majority_smooth(g, 25)),
        ("majority vote +/-2 s on perfect", lambda g, r: majority_smooth(g, 50)),
        ("majority vote +/-5 s on perfect", lambda g, r: majority_smooth(g, 125)),
    ]


def main():
    exps = experiments()
    scores = {name: [] for name, _ in exps}
    stats = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        gt = synthetic_match(rng)
        runs = pm.valid_runs(gt)
        st = pm.possession_stats(gt, runs)
        n_dead = sum(1 for _, _, c in segs(gt, runs) if c == pm.D)
        stats.append((st["n_spells"], st["mean_duration_s"], n_dead))
        for name, fn in exps:
            scores[name].append(spell_f1(gt, fn(gt, np.random.RandomState(1000 + seed))))

    s = np.array(stats, dtype=float).mean(axis=0)
    print("synthetic matches: {} seeds, {:.0f} possessions, mean {:.1f} s, {:.0f} dead stretches\n"
          .format(N_SEEDS, s[0], s[1], s[2]))
    print("{:<36} {:>8} {:>8}".format("damage", "spellF1", "sd"))
    for name, _ in exps:
        v = np.array(scores[name])
        print("{:<36} {:>8.4f} {:>8.4f}".format(name, v.mean(), v.std()))


if __name__ == "__main__":
    main()
