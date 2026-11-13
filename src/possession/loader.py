"""Build a `Match` from the data bundle, once, and cache it.

Parsing the raw ATD (199,319 lines) takes tens of seconds, so the parsed arrays are cached in
`.cache/<game>_match.npz` (gitignored) and keyed on the size and mtime of the source files. Delete
the cache, or pass `rebuild=True`, to force a re-parse.

File-format knowledge is reused from the organiser code (games.py, starter/), never copied:
  * paths and frame counts come from `games`
  * team of each track slot comes from `starter.baseline.team_of_each_track`
  * Smart Tagging comes from `starter.load_smart_tagging.load`
"""

import json
import os

import numpy as np

from possession._paths import CACHE_DIR
from possession.match import Match

CACHE_VERSION = 1
FIELD_SEP, GROUP_SEP = ":", ";"
N_FIELDS = 3  # x, y, speed


def parse_atd_raw(raw_path: str, n_frames: int, n_slots: int, pitch: tuple[float, float]):
    """Raw ATD text -> (ball (n, 3), players (n, S, 3)) float64, positions in metres.

    Line format: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed", NaN when missing,
    positions normalised to 0-1. frame_id counts from 1; row i of the output is video frame i.
    """
    length, width = pitch
    ball = np.full((n_frames, N_FIELDS), np.nan)
    players = np.full((n_frames, n_slots, N_FIELDS), np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            ball[frame] = _group(parts[2])
            flat = parts[1].replace(GROUP_SEP, ",").split(",")
            if len(flat) == n_slots * N_FIELDS:
                players[frame] = np.array(flat, dtype=np.float64).reshape(n_slots, N_FIELDS)
            else:
                groups = parts[1].split(GROUP_SEP)[:n_slots]
                for slot, group in enumerate(groups):
                    players[frame, slot] = _group(group)

    for arr in (ball[..., :2], players[..., :2]):
        arr[..., 0] *= length
        arr[..., 1] *= width
    return ball, players


def _group(text: str) -> np.ndarray:
    values = text.strip().split(",")[:N_FIELDS]
    out = np.full(N_FIELDS, np.nan)
    out[: len(values)] = np.array(values, dtype=np.float64)
    return out


def half_mask(game: str, n_frames: int) -> np.ndarray:
    import games

    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in games.half_video_blocks(game):
        half[first : last + 1] = number
    return half


def load_match(game: str | None = None, rebuild: bool = False, verbose: bool = True) -> Match:
    """The `Match` for `game` (default: games.GAME), from cache when it is fresh."""
    import games
    from starter.baseline import team_of_each_track

    game = game or games.GAME
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game)
    n_frames = games.source_frames("tactical", game)
    pitch = tuple(games.GAMES[game]["pitch"])
    key = _cache_key(game, [meta_path, raw_path])
    cache_path = CACHE_DIR / f"{game}_match.npz"

    arrays = None if rebuild else _read_cache(cache_path, key)
    if arrays is None:
        if verbose:
            print(f"parsing {os.path.basename(raw_path)} (cached afterwards) ...")
        team_of_track = team_of_each_track(meta_path)
        ball, players = parse_atd_raw(raw_path, n_frames, len(team_of_track), pitch)
        arrays = {"ball": ball, "players": players, "team_of_track": team_of_track}
        _write_cache(cache_path, key, arrays)

    match = Match(
        game=game,
        fps=int(games.GAMES[game]["fps"]),
        pitch=pitch,
        ball=arrays["ball"],
        players=arrays["players"],
        team_of_track=arrays["team_of_track"].astype(np.int8),
        half=half_mask(game, n_frames),
        tagging=_load_tagging(game),
    )
    match.validate()
    return match


def _load_tagging(game: str):
    try:
        from starter.load_smart_tagging import load
    except ImportError:
        return None
    try:
        return load(game)
    except OSError:  # games.require raises IOError when the file is absent
        return None


def _cache_key(game: str, paths: list[str]) -> str:
    stats = [(os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in paths]
    return json.dumps({"v": CACHE_VERSION, "game": game, "files": stats})


def _read_cache(path, key):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["key"]) != key:
            return None
        return {name: data[name] for name in ("ball", "players", "team_of_track")}


def _write_cache(path, key, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, key=np.array(key), **arrays)
    os.replace(tmp, path)
