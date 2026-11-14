"""Is IDSSE a fit for training a BAR-ATH possession model? Print the evidence.

    python -m solution.dataset_fit

Three blocks:
  1. inputs   -- frame rate, objects per frame, ball coverage, coordinates
  2. labels   -- the H/A/D structure, measured with the OFFICIAL possession_metrics functions
  3. 5-minute windows -- the same statistics on 5-minute slices of IDSSE, next to the public window,
     to show (a) the label definitions behave alike and (b) how much a 5-minute window scatters.

Reading the public window here is allowed use: comparing *definitions* (CHALLENGE.md: "use it to
understand what is expected"). Nothing below is fitted to it.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import container as C  # noqa: E402
from solution import idsse  # noqa: E402

pm = C.pm

# Params
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
WINDOW = 5 * 60 * C.FPS  # 5 minutes of play, the size of the public window
SHORT = (1.0, 2.0)  # seconds; possessions shorter than this count as "short"


def label_stats(codes):
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    segs = pm.segments(codes, runs)
    dead = [(b - a) / C.FPS for a, b, k in segs if k == C.D]
    dur = np.array([(b - a) / C.FPS for a, b, _ in spells]) if spells else np.zeros(0)
    graded = codes[codes != C.X]
    alive_min = np.sum((graded == C.H) | (graded == C.A)) / C.FPS / 60.0
    return {
        "graded_min": len(graded) / C.FPS / 60.0,
        "share_H": np.mean(graded == C.H), "share_A": np.mean(graded == C.A),
        "share_D": np.mean(graded == C.D),
        "n_poss": len(spells), "poss_per_alive_min": len(spells) / max(alive_min, 1e-9),
        "mean_poss_s": dur.mean() if len(dur) else np.nan,
        "median_poss_s": np.median(dur) if len(dur) else np.nan,
        "lt1s": np.mean(dur < SHORT[0]) if len(dur) else np.nan,
        "lt2s": np.mean(dur < SHORT[1]) if len(dur) else np.nan,
        "n_dead": len(dead), "mean_dead_s": np.mean(dead) if dead else np.nan,
    }


def input_stats(m):
    play = m["half"] > 0
    seen = ~np.isnan(m["px"][play])
    per_team = [seen[:, m["team"] == t].sum(1) for t in (0, 1)]
    ball = ~np.isnan(m["bx"][play])
    L, W = m["pitch"]
    bx, by = m["bx"][play][ball], m["by"][play][ball]
    return {
        "players_mean": float((per_team[0] + per_team[1]).mean()),
        "players_min_max": (int((per_team[0] + per_team[1]).min()),
                            int((per_team[0] + per_team[1]).max())),
        "slots": len(m["team"]),
        "ball_seen": float(ball.mean()),
        "ball_x_range": (float(bx.min()), float(bx.max())),
        "ball_y_range": (float(by.min()), float(by.max())),
        "pitch": (float(L), float(W)),
    }


def fmt_row(name, s):
    return ("{:<16} {:>6.1f} {:>5.0%} {:>5.0%} {:>5.0%} {:>6d} {:>8.2f} {:>8.1f} {:>8.1f}"
            " {:>5.0%} {:>5.0%} {:>6d} {:>7.1f}").format(
        name, s["graded_min"], s["share_H"], s["share_A"], s["share_D"], s["n_poss"],
        s["poss_per_alive_min"], s["mean_poss_s"], s["median_poss_s"], s["lt1s"], s["lt2s"],
        s["n_dead"], s["mean_dead_s"])


HEADER = ("{:<16} {:>6} {:>5} {:>5} {:>5} {:>6} {:>8} {:>8} {:>8} {:>5} {:>5} {:>6} {:>7}".format(
    "", "min", "H", "A", "D", "poss", "poss/min", "mean s", "med s", "<1s", "<2s",
    "dead", "dead s"))


def main():
    matches = idsse.load_all()
    public = pm.label_codes(pm.read_label_csv(PUBLIC_GT))

    print("=" * 100)
    print("1. INPUTS  (BAR-ATH tactical ATD figures are from DATA.md; the match data is gated)")
    print("=" * 100)
    print("{:<16} {:>8} {:>10} {:>7} {:>10} {:>26}".format(
        "", "players", "min/max", "slots", "ball seen", "ball x / y range (m)"))
    for m in matches:
        s = input_stats(m)
        print("{:<16} {:>8.1f} {:>10} {:>7d} {:>10.1%}   x {:>5.1f}..{:<5.1f} y {:>5.1f}..{:<5.1f}".format(
            m["name"], s["players_mean"], "{}/{}".format(*s["players_min_max"]), s["slots"],
            s["ball_seen"], *s["ball_x_range"], *s["ball_y_range"]))
    print("{:<16} {:>8} {:>10} {:>7d} {:>10}".format("BAR-ATH ATD", "20.5", "n/a", 46, "59.3%"))
    print("\nboth: 25 fps, 105 x 68 m pitch, team per object, ball (x, y). IDSSE adds z and person ids,"
          "\nwhich the pipeline deliberately ignores; ATD is normalised 0-1 -> multiply by (L, W).")

    print("\n" + "=" * 100)
    print("2. LABELS  (official possession_metrics: spells = maximal H/A runs, D = stoppage)")
    print("=" * 100)
    print(HEADER)
    full = []
    for m in matches:
        s = label_stats(m["label"])
        full.append(s)
        print(fmt_row(m["name"], s))
    keys = ("poss_per_alive_min", "mean_poss_s", "median_poss_s", "lt1s", "lt2s", "mean_dead_s")
    print("{:<16} ".format("IDSSE mean") + "  ".join(
        "{}={:.2f}".format(k, np.nanmean([s[k] for s in full])) for k in keys))

    print("\n" + "=" * 100)
    print("3. 5-MINUTE WINDOWS  (first 5 min of each IDSSE half, vs the public BAR-ATH window)")
    print("=" * 100)
    print(HEADER)
    windows = []
    for m in matches:
        for half in (1, 2):
            start = int(np.flatnonzero(m["half"] == half)[0])
            codes = np.full_like(m["label"], C.X)
            codes[start:start + WINDOW] = m["label"][start:start + WINDOW]
            s = label_stats(codes)
            windows.append(s)
            print(fmt_row("{} h{}".format(m["name"][-6:], half), s))
    print(fmt_row("BAR-ATH public", label_stats(public)))
    arr = np.array([[s["n_poss"], s["poss_per_alive_min"], s["mean_poss_s"]] for s in windows])
    print("\nIDSSE 5-min windows: possessions {:.0f}-{:.0f} (mean {:.1f}), poss/alive-min {:.1f}-{:.1f},"
          " mean duration {:.1f}-{:.1f} s".format(arr[:, 0].min(), arr[:, 0].max(), arr[:, 0].mean(),
                                                  arr[:, 1].min(), arr[:, 1].max(),
                                                  arr[:, 2].min(), arr[:, 2].max()))


if __name__ == "__main__":
    main()
