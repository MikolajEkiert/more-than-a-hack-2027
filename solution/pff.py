"""PFF FC (Gradient Sports) FIFA World Cup 2022 -> match containers at 25 fps.

64 matches of broadcast-derived tracking with an event stream merged into the frames. Free, but
behind a form: https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset . Unpack it into
data/external/pff/ (gitignored); any layout works as long as the tracking files end in
.jsonl.bz2 (or .jsonl) and the metadata / roster JSONs sit in folders whose names contain "meta"
and "roster".

    python -m solution.pff            # convert everything found, print a summary per match

Labels follow kloppy's PFF reader exactly: game_event_type OUT / END -> dead; FIRSTKICKOFF ...
FOURTHKICKOFF and OTB (on-the-ball event) -> alive; home_ball -> owning team; both carried forward
between events. So, unlike IDSSE, possession changes only at recorded on-ball events -- closer to a
"controlled possession" definition, which is the point of adding it.

Only periods 1 and 2 are kept (extra time would need a third orientation; it is a few matches).
"""

import bz2
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import container as C  # noqa: E402

# Params
RAW_DIR = os.environ.get("MTAH_PFF_DIR") or os.path.join(ROOT, "data", "external", "pff")
NPZ_DIR = os.path.join(RAW_DIR, "npz")
PERIODS = (1, 2)
ALIVE = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"}
DEAD = {"OUT", "END"}
MAX_GAP_S = 0.25         # do not interpolate a position across a hole longer than this


def find_games(raw_dir=RAW_DIR):
    """{game_id: {"tracking", "meta", "roster"}} for every tracking file with its metadata."""
    files = glob.glob(os.path.join(raw_dir, "**", "*.json*"), recursive=True)
    games = {}
    for path in files:
        name = os.path.basename(path)
        gid = name.split(".")[0]
        low = path.lower().replace("\\", "/")
        if name.endswith((".jsonl.bz2", ".jsonl")):
            games.setdefault(gid, {})["tracking"] = path
        elif name.endswith(".json") and "/npz/" not in low:
            folder = os.path.basename(os.path.dirname(low))
            if "meta" in folder:
                games.setdefault(gid, {})["meta"] = path
            elif "roster" in folder:
                games.setdefault(gid, {})["roster"] = path
    return {g: v for g, v in sorted(games.items()) if "tracking" in v and "meta" in v}


def read_meta(path):
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = meta["stadium"]["pitches"][0]
    return {"id": str(meta["id"]), "home": meta["homeTeam"]["name"], "away": meta["awayTeam"]["name"],
            "fps": float(meta["fps"]), "pitch": (float(pitch["length"]), float(pitch["width"]))}


def _open(path):
    return bz2.open(path, "rt", encoding="utf-8") if path.endswith(".bz2") else open(path, encoding="utf-8")


def parse_tracking(path, fps):
    """Per period: time (s), ball x/y, {slot: (x, y)} rows and labels, at the source rate."""
    slots, teams = {}, []
    periods = {p: {"t": [], "bx": [], "by": [], "players": [], "label": []} for p in PERIODS}
    alive, owner = False, None
    with _open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            event = frame.get("game_event") or {}
            kind = event.get("game_event_type")
            if kind in DEAD:
                alive = False
            elif kind in ALIVE:
                alive = True
            if event.get("home_ball") is not None:
                owner = 0 if event["home_ball"] else 1
            period = frame.get("period")
            if period not in periods:
                continue
            store = periods[period]
            t = frame.get("periodElapsedTime")
            store["t"].append(float(t) if t is not None else frame["frameNum"] / fps)
            ball = frame.get("ballsSmoothed") or {}
            store["bx"].append(np.nan if ball.get("x") is None else float(ball["x"]))
            store["by"].append(np.nan if ball.get("y") is None else float(ball["y"]))
            row = {}
            for side, key in ((0, "homePlayersSmoothed"), (1, "awayPlayersSmoothed")):
                for p in frame.get(key) or []:
                    if p.get("x") is None or p.get("y") is None or p.get("jerseyNum") is None:
                        continue
                    ident = (side, int(p["jerseyNum"]))
                    if ident not in slots:
                        slots[ident] = len(teams)
                        teams.append(side)
                    row[slots[ident]] = (float(p["x"]), float(p["y"]))
            store["players"].append(row)
            store["label"].append(C.D if not alive else (C.X if owner is None else (C.H if owner == 0 else C.A)))
    return periods, np.array(teams, np.int8)


def _resample(t, values, grid):
    """Linear interpolation onto `grid`; NaN where a neighbour is NaN or the gap is too long."""
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    dt = t[i + 1] - t[i]
    w = np.clip((grid - t[i]) / np.where(dt > 0, dt, 1), 0, 1)
    if values.ndim == 2:
        w = w[:, None]
    out = values[i] * (1 - w) + values[i + 1] * w
    out[dt > MAX_GAP_S] = np.nan
    return out


def build_container(game, raw_dir=RAW_DIR):
    meta = read_meta(game["meta"])
    periods, team = parse_tracking(game["tracking"], meta["fps"])
    length, width = meta["pitch"]
    n_slots = len(team)
    halves = []
    for p in PERIODS:
        s = periods[p]
        if len(s["t"]) < 2:
            continue
        t = np.asarray(s["t"])
        order = np.argsort(t, kind="stable")
        t = t[order]
        px = np.full((len(t), n_slots), np.nan)
        py = np.full((len(t), n_slots), np.nan)
        for r, k in enumerate(order):
            for slot, (x, y) in s["players"][k].items():
                px[r, slot], py[r, slot] = x, y
        bx, by = np.asarray(s["bx"])[order], np.asarray(s["by"])[order]
        label = np.asarray(s["label"], np.int8)[order]
        grid = np.arange(t[0], t[-1], 1.0 / C.FPS)
        idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        halves.append({"px": _resample(t, px, grid).astype(np.float32),
                       "py": _resample(t, py, grid).astype(np.float32),
                       "bx": _resample(t, bx, grid).astype(np.float32),
                       "by": _resample(t, by, grid).astype(np.float32),
                       "label": label[idx]})
    match = C.concat_halves(halves, team, (length, width), "PFF-" + meta["id"])
    # PFF coordinates are metres from the centre spot; the container wants [0, L] x [0, W]
    if np.nanmin(match["px"]) < -5 or np.nanmin(match["bx"]) < -5:
        for k, off in (("px", length / 2), ("bx", length / 2), ("py", width / 2), ("by", width / 2)):
            match[k] = (match[k] + off).astype(np.float32)
    match["_meta"] = meta
    return match


def convert_all(raw_dir=RAW_DIR, npz_dir=NPZ_DIR, log=print):
    games = find_games(raw_dir)
    log("{} PFF games with tracking + metadata under {}".format(len(games), raw_dir))
    for gid, game in games.items():
        path = os.path.join(npz_dir, gid + ".npz")
        if os.path.isfile(path):
            continue
        m = build_container(game, raw_dir)
        C.save(m, path)
        play = m["half"] > 0
        lab = m["label"][play]
        log("  {} {} v {} | frames {} | slots {} | ball seen {:.0%} | H {:.0%} A {:.0%} D {:.0%} X {:.0%}".format(
            gid, m["_meta"]["home"], m["_meta"]["away"], int(play.sum()), len(m["team"]),
            float(np.mean(~np.isnan(m["bx"][play]))), *(float(np.mean(lab == c)) for c in (C.H, C.A, C.D, C.X))))
    return sorted(glob.glob(os.path.join(npz_dir, "*.npz")))


def loaders(npz_dir=NPZ_DIR, limit=None):
    """One zero-argument loader per converted match -- 64 containers do not fit in RAM at once."""
    paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))[:limit]
    return [(lambda p=p: C.load(p)) for p in paths]


def available(raw_dir=RAW_DIR):
    return bool(find_games(raw_dir)) or bool(glob.glob(os.path.join(NPZ_DIR, "*.npz")))


if __name__ == "__main__":
    convert_all()
