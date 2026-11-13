"""IDSSE / Sportec open data (Bassek et al., Scientific Data 2025) -> match containers.

Seven DFL matches, 2022/23, TRACAB optical tracking at 25 Hz, CC-BY 4.0. What makes it the
training set for this challenge: every ball frame carries

    BallStatus     1 = alive, 0 = dead           (set by the provider on corrected tracking)
    BallPossession 1 = home,  2 = away           (the team the corrected tracking names as owner)

which is exactly how the challenge ground truth is built (CHALLENGE.md: "a frame is D where the
corrected tracking marks the ball dead, and otherwise takes the team the tracking names as owning
it"). So the labels below are derived by the same mechanical rule, with no event-data heuristics.

    python -m solution.idsse                 # download (if missing) + convert all seven, print summary

Raw XML goes to data/external/idsse/ (gitignored), containers to data/external/idsse/npz/.
"""

import argparse
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from solution import container as C  # noqa: E402

# Params
RAW_DIR = os.path.join(ROOT, "data", "external", "idsse")
NPZ_DIR = os.path.join(RAW_DIR, "npz")
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = {  # match id -> competition id (1 = Bundesliga, 2 = 2. Bundesliga)
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002", "J03WR9": "DFL-COM-000002",
}
SECTIONS = ("firstHalf", "secondHalf")


def filenames(match_id):
    comp = MATCHES[match_id]
    return {
        "meta": "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(comp, match_id),
        "events": "DFL_03_02_events_raw_{}_DFL-MAT-{}.xml".format(comp, match_id),
        "positions": "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(comp, match_id),
    }


def download(match_id, raw_dir=RAW_DIR):
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for kind, name in filenames(match_id).items():
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            print("  downloading", name)
            urllib.request.urlretrieve(BASE_URL + name, path)
        paths[kind] = path
    return paths


def read_meta(meta_path):
    general = ET.parse(meta_path).getroot().find(".//General")
    env = ET.parse(meta_path).getroot().find(".//Environment")
    return {
        "home": general.get("HomeTeamId"), "away": general.get("GuestTeamId"),
        "title": general.get("MatchTitle"), "competition": general.get("CompetitionName"),
        "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))),
    }


def _attr_index(tokens):
    """Attribute name -> token index of its value, from one '"'-split XML element line."""
    return {tokens[i].strip().rstrip("=").split()[-1]: i + 1
            for i in range(0, len(tokens) - 1, 2)}


def parse_positions(pos_path):
    """Stream the 350-420 MB XML once. Returns {section: {person: (N, X, Y)}, ball: {...}}.

    Line-split rather than an XML parser: every <Frame/> sits on its own line with a fixed
    attribute order inside a FrameSet, which makes this ~10x faster than ElementTree.
    """
    tracks = {s: {} for s in SECTIONS}
    ball = {s: None for s in SECTIONS}
    section = person = team = None
    rows, index = [], None

    def flush():
        if section is None or not rows:
            return
        arr = np.array(rows, dtype=np.float64)
        if team == "BALL":
            ball[section] = arr  # N, X, Y, Z, BallPossession, BallStatus
        else:
            tracks[section][(team, person)] = arr  # N, X, Y

    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                tokens = line.split('"')
                if index is None:
                    index = _attr_index(tokens)
                    keys = ["N", "X", "Y"] + (["Z", "BallPossession", "BallStatus"]
                                               if team == "BALL" else [])
                    cols = [index[k] for k in keys]
                rows.append([float(tokens[c]) for c in cols])
            elif line.startswith("<FrameSet "):
                flush()
                tokens = line.split('"')
                attrs = _attr_index(tokens)
                section = tokens[attrs["GameSection"]]
                team = tokens[attrs["TeamId"]]
                person = tokens[attrs["PersonId"]]
                rows, index = [], None
        flush()
    return tracks, ball


def build_container(match_id, raw_dir=RAW_DIR):
    paths = download(match_id, raw_dir)
    meta = read_meta(paths["meta"])
    length, width = meta["pitch"]
    tracks, ball = parse_positions(paths["positions"])

    # Players only: the Bundesliga files also carry TeamId="referee" (referee + assistants), which
    # the ATD does not have and which would otherwise land in a team as three phantom players.
    clubs = (meta["home"], meta["away"])
    people = sorted({key for s in SECTIONS for key in tracks[s] if key[0] in clubs},
                    key=lambda k: (k[0] != meta["home"], k[1]))
    team = [0 if t == meta["home"] else 1 for t, _ in people]
    slot = {key: i for i, key in enumerate(people)}

    halves, stats = [], {}
    for s in SECTIONS:
        b = ball[s]
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        t = n1 - n0 + 1
        px = np.full((t, len(people)), np.nan, np.float32)
        py = np.full((t, len(people)), np.nan, np.float32)
        for key, arr in tracks[s].items():
            if key not in slot:
                continue
            idx = arr[:, 0].astype(np.int64) - n0
            ok = (idx >= 0) & (idx < t)
            px[idx[ok], slot[key]] = arr[ok, 1] + length / 2.0
            py[idx[ok], slot[key]] = arr[ok, 2] + width / 2.0

        bx = np.full(t, np.nan, np.float32)
        by = np.full(t, np.nan, np.float32)
        poss = np.zeros(t, np.int8)
        status = np.zeros(t, np.int8)
        idx = b[:, 0].astype(np.int64) - n0
        bx[idx] = b[:, 1] + length / 2.0
        by[idx] = b[:, 2] + width / 2.0
        poss[idx] = b[:, 4].astype(np.int8)
        status[idx] = b[:, 5].astype(np.int8)

        # The challenge's rule, verbatim: dead -> D, otherwise the owning team.
        label = np.full(t, C.D, np.int8)
        alive = status == 1
        label[alive & (poss == 1)] = C.H
        label[alive & (poss == 2)] = C.A
        stats[s] = {"frames": t, "alive_without_owner": int(np.sum(alive & ~np.isin(poss, (1, 2))))}
        halves.append({"px": px, "py": py, "bx": bx, "by": by, "label": label})

    match = C.concat_halves(halves, team, (length, width), "IDSSE-" + match_id)
    match["_meta"] = meta
    match["_stats"] = stats
    return match


def load_all(npz_dir=NPZ_DIR, match_ids=None):
    """Containers for the requested (default: all) matches, converting any that are missing."""
    out = []
    for match_id in (match_ids or sorted(MATCHES)):
        path = os.path.join(npz_dir, match_id + ".npz")
        if not os.path.isfile(path):
            convert(match_id, npz_dir)
        out.append(C.load(path))
    return out


def convert(match_id, npz_dir=NPZ_DIR):
    print("converting", match_id)
    match = build_container(match_id)
    C.save(match, os.path.join(npz_dir, match_id + ".npz"))
    print("  {} | {} | pitch {} | slots {} | frames {} | alive-without-owner {}".format(
        match["_meta"]["competition"], match["_meta"]["title"], match["_meta"]["pitch"],
        len(match["team"]), len(match["label"]),
        sum(v["alive_without_owner"] for v in match["_stats"].values())))
    return match


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=sorted(MATCHES))
    parser.add_argument("--force", action="store_true", help="reconvert even if the npz exists")
    args = parser.parse_args()
    for m in args.matches:
        if args.force or not os.path.isfile(os.path.join(NPZ_DIR, m + ".npz")):
            convert(m)
