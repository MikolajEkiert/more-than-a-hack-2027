"""BAR-ATH tactical ATD -> match container (the same layout IDSSE gets, see container.py).

    python -m solution.atd          # convert, cache to data/bar_ath/BAR-ATH.npz, print the checks

Raw line format (see starter/baseline.py): "<frame_id>:<slot>;<slot>;...:<ball>", each group
"x,y,speed" normalised 0-1, "NaN" when unseen. video_frame = frame_id - 1. Team per slot comes
from the metadata XML (first team = home = Barcelona). Labels are all X: the match is unlabelled
for training purposes. The public window is attached separately and only for alignment checks.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from solution import container as C  # noqa: E402

# Params
CACHE = os.path.join(ROOT, "data", "bar_ath", "BAR-ATH.npz")
EXPECTED = {"frames": 199319, "ball_seen_all": 0.569, "ball_seen_graded": 0.593,
            "players_graded": 20.5}  # DATA.md


def have_data(game=GAME):
    try:
        require(["atd_meta", "atd_raw"], game)
        return True
    except IOError:
        return False


def parse(game=GAME):
    import baseline  # starter/baseline.py: team_of_each_track() reads the slot -> team map
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = baseline.team_of_each_track(meta_path)
    n_frames, n_slots = source_frames("tactical", game), len(team)
    length, width = GAMES[game]["pitch"]

    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
            if len(vals) != 3 * n_slots:
                raise ValueError("frame {}: {} values, expected {}".format(
                    parts[0], len(vals), 3 * n_slots))
            px[row] = vals[0::3] * length
            py[row] = vals[1::3] * width
            ball = parts[2].split(",")
            bx[row], by[row] = float(ball[0]) * length, float(ball[1]) * width

    half = np.zeros(n_frames, np.int8)
    for h, first, last in half_video_blocks(game):
        half[first:last + 1] = h
    return {"px": px, "py": py, "team": team.astype(np.int8), "bx": bx, "by": by,
            "half": half, "label": np.full(n_frames, C.X, np.int8),
            "pitch": np.array((length, width), np.float32), "name": game}


def load(game=GAME, refresh=False):
    if not refresh and os.path.isfile(CACHE):
        return C.load(CACHE)
    match = parse(game)
    C.save(match, CACHE)
    return match


def checks(m):
    """The published figures, recomputed. If these disagree, the load is wrong -- stop."""
    graded = m["half"] > 0
    seen = ~np.isnan(m["bx"])
    out = {
        "frames": len(m["bx"]),
        "ball_seen_all": float(seen.mean()),
        "ball_seen_graded": float(seen[graded].mean()),
        "players_graded": float((~np.isnan(m["px"][graded])).sum(1).mean()),
    }
    out["ok"] = True
    for key, want in EXPECTED.items():
        got = out[key]
        ok = got == want if key == "frames" else abs(got - want) < 0.01 * max(1, want)
        out["ok"] &= ok
        print("  {:<18} {:>10.3f}  expected {:>8}  {}".format(key, got, want, "ok" if ok else "MISMATCH"))
    return out


if __name__ == "__main__":
    match = load(refresh=True)
    if not checks(match)["ok"]:
        sys.exit("load does not match DATA.md -- fix before going further")
