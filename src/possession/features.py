"""Shared derived quantities, computed once per match and cached on it.

Before writing a helper inside a method, check whether it belongs here. Anything two methods could
both use (distances, velocities, team shapes, tagging masks) lives in this module, with a test.
Every function takes the `Match` first and memoises on `match._cache`.
"""

import numpy as np

from possession.match import Match

CHUNK = 20_000  # frames per block, bounds memory for (n, S) intermediates


def _memo(match: Match, key, compute):
    if key not in match._cache:
        match._cache[key] = compute()
    return match._cache[key]


def nearest_player(match: Match) -> tuple[np.ndarray, np.ndarray]:
    """(distance_m, team) of the tracked player nearest the ball, per frame.

    distance is NaN and team is -1 where the ball or every player is missing. Ties go to the lowest
    slot index, matching starter/baseline.py exactly.
    """

    def compute():
        n = match.n_frames
        distance = np.full(n, np.nan)
        slot = np.full(n, -1, dtype=np.int64)
        for start in range(0, n, CHUNK):
            stop = min(n, start + CHUNK)
            ball = match.ball[start:stop, None, :2]
            delta = match.players[start:stop, :, :2] - ball
            squared = delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
            squared = np.where(np.isnan(squared), np.inf, squared)
            best = np.argmin(squared, axis=1)
            best_sq = squared[np.arange(stop - start), best]
            found = np.isfinite(best_sq)
            distance[start:stop][found] = np.sqrt(best_sq[found])
            slot[start:stop][found] = best[found]
        team = np.where(slot >= 0, match.team_of_track[np.maximum(slot, 0)], -1).astype(np.int8)
        return distance, team

    return _memo(match, "nearest_player", compute)


def tagging_mask(match: Match, code: str | None = None, **tags: str) -> np.ndarray:
    """Boolean (n,) mask of frames covered by Smart Tagging instances matching `code` and `tags`.

    Matching is case-insensitive (DATA.md: codes are upper case, tag values title case). Returns all
    False when the match has no tagging. Example: tagging_mask(m, "SET PIECES", Type="Throw-in").
    """

    def compute():
        mask = np.zeros(match.n_frames, dtype=bool)
        for item in match.tagging or []:
            if code and item["code"].lower() != code.lower():
                continue
            values = {k.lower(): (v or "").lower() for k, v in item["tags"].items()}
            if any(values.get(k.lower()) != v.lower() for k, v in tags.items()):
                continue
            mask[max(0, item["start_frame"]) : max(0, item["end_frame"])] = True
        return mask

    key = ("tagging_mask", code, tuple(sorted(tags.items())))
    return _memo(match, key, compute)
