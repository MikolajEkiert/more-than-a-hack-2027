"""SkillCorner open data (A-League 2024/25, broadcast tracking at 10 Hz) as package containers.

Task 17 adds other providers' matches to the training of the team-feature model; the idea comes
from other teams (research/COMPETITORS.md section 1). This module downloads the 20 matches of
https://github.com/SkillCorner/opendata (MIT licence) at one fixed commit, checks every file
against `skillcorner.sha256` (the git-LFS sha256 of each tracking file, the hash of each metadata
file at that commit), and converts each match to the layout of solution/container.py at 25 Hz.
Nothing is redistributed; files live under .cache/skillcorner/.

Labels, from the provider's per-frame `possession.group` (the team of the player SkillCorner
names in possession, carried while the ball travels between players): "home team" -> H,
"away team" -> A, none -> D. On the open matches the frames without a group lie in long runs
(median about 17 s) that start and end at the interruptions of the provider's event file (about
eight in ten within 3 s), so they read as dead ball. The tracking is broadcast tracking: only the
players in or near the picture are listed, and the ball is kept only where SkillCorner detected
it (the file also carries extrapolated ball positions, which are guesses). Frames with almost no
players listed (replays, close-ups: half of the dead time) are labelled X, so a model never learns
that an empty picture means a dead ball, which the competition tracking never shows. Training
degrades these matches exactly as the PFF ones (`supervised._extra_match_rows`): the ball loss is
topped up to the competition match's, counting what is already missing.
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from possession._paths import CACHE_DIR

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"  # SkillCorner/opendata, 2026-09-14
RAW_URL = f"https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}/data/matches/"
LFS_URL = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{COMMIT}/data/matches/"
HASHES = Path(__file__).with_name("skillcorner.sha256")
SC_DIR = CACHE_DIR / "skillcorner"
RAW_DIR = SC_DIR / "raw"
NPZ_DIR = SC_DIR / "npz"
SOURCE_FPS = 10
PERIODS = (1, 2)
MAX_GAP_S = 0.15  # no interpolation across a missing 10 Hz frame
# A frame with fewer players than this is a broadcast picture without tracking (a replay, a
# close-up): it is labelled X, not trained on. Half of the dead time is such frames, 2% of play.
MIN_PLAYERS = 6
GROUPS = {"home team": 0, "away team": 1}  # possession_metrics codes H = 0, A = 1


def expected_hashes() -> dict[str, str]:
    """{file name: sha256} of every file at the pinned commit."""
    return dict(line.split()[::-1] for line in HASHES.read_text().splitlines() if line.strip())


def match_ids() -> tuple[str, ...]:
    """The 20 match ids of the open data, sorted."""
    names = expected_hashes()
    return tuple(sorted({name.split("_")[0] for name in names}))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(match_id: str, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Paths of the match's tracking and metadata files, downloading what is missing. A download
    lands in "<name>.part" and is renamed only once its sha256 matches the listing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_hashes()
    paths = {}
    for kind, name, base in (
        ("tracking", f"{match_id}_tracking_extrapolated.jsonl", LFS_URL),
        ("meta", f"{match_id}_match.json", RAW_URL),
    ):
        path = raw_dir / name
        if not path.is_file() or _sha256(path) != expected[name]:
            partial = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(f"{base}{match_id}/{name}") as response:  # noqa: S310
                partial.write_bytes(response.read())
            if _sha256(partial) != expected[name]:
                partial.unlink()
                raise ValueError(f"{name}: sha256 does not match {HASHES.name}")
            partial.replace(path)
        paths[kind] = path
    return paths


def _resample(t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto `grid`; NaN where a neighbour is NaN or the source has a gap."""
    i = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    dt = t[i + 1] - t[i]
    w = np.clip((grid - t[i]) / np.where(dt > 0, dt, 1), 0, 1)
    if values.ndim == 2:
        w = w[:, None]
    out = values[i] * (1 - w) + values[i + 1] * w
    out[dt > MAX_GAP_S] = np.nan
    return out


def parse(tracking: Path, meta: Path) -> dict:
    """The match as a package container at 25 Hz (positions in metres, [0, L] x [0, W])."""
    from solution import container as package_container

    info = json.loads(Path(meta).read_text(encoding="utf-8"))
    home = info["home_team"]["id"]
    length, width = float(info["pitch_length"]), float(info["pitch_width"])
    slot_of = {}
    teams = []
    for player in sorted(info["players"], key=lambda p: p["id"]):
        slot_of[player["id"]] = len(teams)
        teams.append(0 if player["team_id"] == home else 1)
    n_slots = len(teams)
    periods = {p: {"frame": [], "bx": [], "by": [], "players": [], "label": []} for p in PERIODS}
    with open(tracking, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            store = periods.get(frame.get("period"))
            if store is None:
                continue
            store["frame"].append(int(frame["frame"]))
            ball = frame.get("ball_data") or {}
            seen = ball.get("is_detected") and ball.get("x") is not None
            store["bx"].append(float(ball["x"]) + length / 2 if seen else np.nan)
            store["by"].append(float(ball["y"]) + width / 2 if seen else np.nan)
            row = {}
            for p in frame.get("player_data") or []:
                slot = slot_of.get(p.get("player_id"))
                if slot is not None and p.get("x") is not None and p.get("y") is not None:
                    row[slot] = (float(p["x"]) + length / 2, float(p["y"]) + width / 2)
            store["players"].append(row)
            group = (frame.get("possession") or {}).get("group")
            store["label"].append(GROUPS.get(group, package_container.D))
    halves = []
    for p in PERIODS:
        s = periods[p]
        if len(s["frame"]) < 2:
            continue
        order = np.argsort(np.asarray(s["frame"]), kind="stable")
        t = np.asarray(s["frame"], float)[order] / SOURCE_FPS
        px = np.full((len(t), n_slots), np.nan)
        py = np.full((len(t), n_slots), np.nan)
        for r, k in enumerate(order):
            for slot, (x, y) in s["players"][k].items():
                px[r, slot], py[r, slot] = x, y
        bx, by = np.asarray(s["bx"])[order], np.asarray(s["by"])[order]
        label = np.asarray(s["label"], np.int8)[order]
        grid = np.arange(t[0], t[-1], 1.0 / package_container.FPS)
        idx = np.clip(np.searchsorted(t, grid + 1e-9, side="right") - 1, 0, len(t) - 1)
        half = {
            "px": _resample(t, px, grid).astype(np.float32),
            "py": _resample(t, py, grid).astype(np.float32),
            "bx": _resample(t, bx, grid).astype(np.float32),
            "by": _resample(t, by, grid).astype(np.float32),
            "label": label[idx],
        }
        untracked = np.sum(~np.isnan(half["px"]), axis=1) < MIN_PLAYERS
        half["label"] = np.where(untracked, package_container.X, half["label"]).astype(np.int8)
        halves.append(half)
    return package_container.concat_halves(
        halves, np.asarray(teams, np.int8), (length, width), f"SC-{info['id']}"
    )


def convert(match_id: str) -> Path:
    """Download (if needed) and convert one match; returns its container file."""
    from solution import container as package_container

    path = NPZ_DIR / f"{match_id}.npz"
    if not path.is_file():
        files = download(match_id)
        m = parse(files["tracking"], files["meta"])
        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        partial = NPZ_DIR / f"{match_id}.{os.getpid()}.part.npz"
        package_container.save(m, str(partial))
        partial.replace(path)
    return path


def container_files() -> list[Path]:
    """The container file of every match, converting what is missing."""
    return [convert(match_id) for match_id in match_ids()]


def loaders() -> list:
    """One zero-argument loader per match, in `match_ids` order."""
    from solution import container as package_container

    return [(lambda p=str(path): package_container.load(p)) for path in container_files()]
