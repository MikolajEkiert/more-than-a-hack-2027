"""Why does a model that works on IDSSE fail on BAR-ATH? Compare inputs, print verdicts only.

    python -m solution.diagnose_shift

Output is pass/fail per check plus the NAMES of features whose BAR-ATH values fall outside the
IDSSE range -- no values, no statistics of the match. That keeps it shareable with a cloud
assistant under the competition terms (no extracts of the dataset).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import atd, container as C, degrade, features, idsse  # noqa: E402

# Params
REFERENCE = "J03WPY"
INSIDE_MIN = 0.80        # a feature is "shifted" if fewer than 80% of BAR-ATH values sit inside
SEVERE = 0.50            # IDSSE's 1-99 % range, "severe" below 50 %


def verdict(name, ok):
    print("  {:<58} {}".format(name, "ok" if ok else "FAIL"))
    return ok


def geometry_checks(m, tag):
    L, W = [float(v) for v in m["pitch"]]
    g = m["half"] > 0
    px, py, bx, by = m["px"][g], m["py"][g], m["bx"][g], m["by"][g]
    seen_p, seen_b = ~np.isnan(px), ~np.isnan(bx)
    inside_p = ((px > -5) & (px < L + 5) & (py > -5) & (py < W + 5))[seen_p].mean()
    inside_b = ((bx > -5) & (bx < L + 5) & (by > -5) & (by < W + 5))[seen_b].mean()
    home = (~np.isnan(px[:, m["team"] == 0])).sum(1)
    away = (~np.isnan(px[:, m["team"] == 1])).sum(1)
    step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))
    moving = step[~np.isnan(step)]
    speed = np.median(moving) * C.FPS
    frozen = np.mean(moving < 1e-6)
    sides = []
    for h in (1, 2):
        sel = m["half"] == h
        sides.append(abs(np.nanmean(m["px"][sel][:, m["team"] == 0]) -
                         np.nanmean(m["px"][sel][:, m["team"] == 1])) > 5)
    print(tag)
    verdict("players inside the pitch (+5 m) on >= 99% of observations", inside_p >= 0.99)
    verdict("ball inside the pitch (+5 m) on >= 95% of observations", inside_b >= 0.95)
    verdict("median visible players per team between 8 and 12", 8 <= np.median(home) <= 12
            and 8 <= np.median(away) <= 12)
    verdict("median player speed between 0.5 and 4 m/s", 0.5 <= speed <= 4)
    verdict("player positions frozen frame-to-frame on < 20% of steps", frozen < 0.2)
    verdict("teams on opposite sides in both halves", all(sides))


def main():
    params_path = os.path.join(ROOT, "data", "models", "degrade_params.json")
    params = json.load(open(params_path)) if os.path.isfile(params_path) else {}
    ref = degrade.degrade(idsse.load_all(match_ids=[REFERENCE])[0], seed=999, **params)
    bar = atd.load()

    geometry_checks(ref, "IDSSE reference (degraded) -- should be all ok:")
    geometry_checks(bar, "BAR-ATH:")

    Fr, names = features.build_features_v1(ref)
    Fb, _ = features.build_features_v1(bar)
    Fr, Fb = Fr[ref["half"] > 0], Fb[bar["half"] > 0]
    shifted, severe, nan_only = [], [], []
    for j, name in enumerate(names):
        r, b = Fr[:, j], Fb[:, j]
        r, b = r[~np.isnan(r)], b[~np.isnan(b)]
        if len(b) == 0:
            nan_only.append(name)
            continue
        lo, hi = np.percentile(r, [1, 99])
        inside = np.mean((b >= lo) & (b <= hi))
        side = "higher" if np.mean(b > hi) > np.mean(b < lo) else "lower"
        if inside < SEVERE:
            severe.append("{} ({})".format(name, side))
        elif inside < INSIDE_MIN:
            shifted.append("{} ({})".format(name, side))
    print("\nfeatures, BAR-ATH vs IDSSE range ({} total):".format(len(names)))
    print("  severely shifted (<50% inside):", ", ".join(severe) or "none")
    print("  shifted (50-80% inside):       ", ", ".join(shifted) or "none")
    print("  always missing on BAR-ATH:     ", ", ".join(nan_only) or "none")


if __name__ == "__main__":
    main()
