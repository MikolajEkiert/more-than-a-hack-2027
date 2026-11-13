"""One in-memory layout for every match, external or ours, so features are written once.

A match is a dict of numpy arrays on its own 25 fps timeline:

    px, py   float32 (T, S)  player positions in metres, pitch frame [0, L] x [0, W], NaN = unseen
    team     int8    (S,)    0 = home, 1 = away (one entry per track slot)
    bx, by   float32 (T,)    ball position in metres, NaN = unseen
    half     int8    (T,)    1 / 2 for match frames, 0 for padding between halves
    label    int8    (T,)    possession_metrics codes H=0 A=1 D=2 X=3 (X = not graded / unknown)
    pitch    float32 (2,)    (L, W) in metres
    name     str

Slots are anonymous on purpose: the provided ATD has 46 tracklet slots with a team and nothing
else, so nothing downstream may rely on a slot being one person for the whole match.

The two halves are separated by HALF_GAP frames of padding labelled X, which is how the official
scorer sees halftime too: valid_runs() splits on X, so no possession can straddle the break.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
HALF_GAP = 250  # padding frames between halves (10 s), labelled X so the scorer never bridges them
H, A, D, X = pm.H, pm.A, pm.D, pm.X

FIELDS = ("px", "py", "team", "bx", "by", "half", "label", "pitch")


def save(match, path):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    np.savez_compressed(path, name=np.array(match["name"]), **{k: match[k] for k in FIELDS})


def load(path):
    with np.load(path, allow_pickle=False) as z:
        match = {k: z[k] for k in FIELDS}
        match["name"] = str(z["name"])
    return match


def concat_halves(halves, team, pitch, name):
    """Stack per-half dicts (px, py, bx, by, label) into one match with X padding between."""
    n_slots = len(team)
    parts = {k: [] for k in ("px", "py", "bx", "by", "half", "label")}
    for i, h in enumerate(halves):
        if i:
            parts["px"].append(np.full((HALF_GAP, n_slots), np.nan, np.float32))
            parts["py"].append(np.full((HALF_GAP, n_slots), np.nan, np.float32))
            parts["bx"].append(np.full(HALF_GAP, np.nan, np.float32))
            parts["by"].append(np.full(HALF_GAP, np.nan, np.float32))
            parts["half"].append(np.zeros(HALF_GAP, np.int8))
            parts["label"].append(np.full(HALF_GAP, X, np.int8))
        t = len(h["label"])
        for k in ("px", "py", "bx", "by", "label"):
            parts[k].append(h[k])
        parts["half"].append(np.full(t, i + 1, np.int8))
    match = {k: np.concatenate(v) for k, v in parts.items()}
    match.update(team=np.asarray(team, np.int8), pitch=np.asarray(pitch, np.float32), name=name)
    return match


def swap_teams(match):
    """Same match with home and away exchanged -- the symmetry augmentation for training."""
    out = dict(match)
    out["team"] = (1 - match["team"]).astype(np.int8)
    label = match["label"].copy()
    label[match["label"] == H] = A
    label[match["label"] == A] = H
    out["label"] = label
    out["name"] = match["name"] + "_swap"
    return out
