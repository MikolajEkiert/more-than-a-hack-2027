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


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, stops) of the True runs of a boolean array; stops are exclusive."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# Savitzky-Golay, quadratic, 7 frames (Vidal-Codina et al. 2022 use the same on ball x, y):
# least-squares value and slope at the centre of the window. Written out because the project
# does not depend on scipy.
_SG_SMOOTH = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0
_SG_SLOPE = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) / 28.0
_SG_HALF = 3


def ball_kinematics(match: Match) -> dict:
    """Ball motion derived from positions: xy, velocity, speed, acceleration, heading change.

    The ATD's own ball speed column is not used: on the real data it does not agree with the
    positions (see docs/DECISIONS.md). Keys, all (n,) or (n, 2) float64, NaN where the ball is
    missing:
      xy        smoothed position, metres
      velocity  metres per second
      speed     |velocity|
      accel     change of speed, m/s^2 (negative when slowing)
      turn_deg  angle between the velocity 3 frames before and 3 frames after, degrees
    Within 3 frames of a gap the smoothing window is incomplete: there xy is the raw position and
    velocity a one-sided difference, so every tracked frame has a value.
    """

    def compute():
        n, fps = match.n_frames, match.fps
        raw = match.ball[:, :2]
        seen = match.ball_seen
        filled = np.where(seen[:, None], raw, 0.0)
        full = np.convolve(seen.astype(np.float64), np.ones(7), mode="same") > 6.5

        xy = np.full((n, 2), np.nan)
        velocity = np.full((n, 2), np.nan)
        for axis in range(2):
            # np.convolve flips its kernel: reversing gives a correlation, i.e. window order.
            smooth = np.convolve(filled[:, axis], _SG_SMOOTH[::-1], mode="same")
            slope = np.convolve(filled[:, axis], _SG_SLOPE[::-1], mode="same") * fps
            xy[:, axis] = np.where(full, smooth, np.where(seen, raw[:, axis], np.nan))
            velocity[:, axis] = np.where(full, slope, np.nan)

        # Edges of a tracked run: fall back to a one-sided difference of the raw positions.
        forward = np.full((n, 2), np.nan)
        forward[:-1] = (raw[1:] - raw[:-1]) * fps
        backward = np.full((n, 2), np.nan)
        backward[1:] = forward[:-1]
        edge = seen & ~full
        fallback = np.where(np.isnan(forward), backward, forward)
        velocity[edge] = fallback[edge]

        speed = np.hypot(velocity[:, 0], velocity[:, 1])
        accel = np.full(n, np.nan)
        accel[1:-1] = (speed[2:] - speed[:-2]) * fps / 2.0

        k = _SG_HALF
        turn = np.full(n, np.nan)
        before, after = velocity[: -2 * k], velocity[2 * k :]
        dot = (before * after).sum(axis=1)
        norm = np.hypot(before[:, 0], before[:, 1]) * np.hypot(after[:, 0], after[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.clip(dot / norm, -1.0, 1.0)
        turn[k:-k] = np.degrees(np.arccos(cos))
        return {"xy": xy, "velocity": velocity, "speed": speed, "accel": accel, "turn_deg": turn}

    return _memo(match, "ball_kinematics", compute)


def ball_edge_distance(match: Match) -> np.ndarray:
    """Distance (m) from the ball to the nearest boundary: negative outside, NaN if missing."""

    def compute():
        length, width = match.pitch
        x, y = match.ball[:, 0], match.ball[:, 1]
        return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))

    return _memo(match, "ball_edge_distance", compute)


def collective_speed(match: Match, percentile: float = 90.0) -> np.ndarray:
    """Per-frame percentile of the tracked players' speeds (m/s); NaN where nobody is tracked.

    A high percentile follows the players who are actually running: during play a few always are,
    during a stoppage almost nobody is.
    """

    def compute():
        n = match.n_frames
        out = np.full(n, np.nan)
        for start in range(0, n, CHUNK):
            block = match.players[start : start + CHUNK, :, 2]
            some = ~np.isnan(block).all(axis=1)
            if some.any():
                out[start : start + CHUNK][some] = np.nanpercentile(block[some], percentile, axis=1)
        return out

    return _memo(match, ("collective_speed", float(percentile)), compute)


def ball_gaps(match: Match) -> dict:
    """Stretches of graded frames where the ball is not tracked, with what surrounds them.

    One entry per gap, never crossing a half. Keys (arrays of equal length):
      start, stop      video frames, stop exclusive
      half             1 or 2
      duration_s
      players_speed    median over the gap of collective_speed (NaN if no players at all)
      edge_before_m    ball_edge_distance at the last tracked frame before the gap (NaN if none)
      edge_after_m     the same at the first tracked frame after it
    """

    def compute():
        seen = match.ball_seen
        speed = collective_speed(match)
        edge = ball_edge_distance(match)
        starts, stops, halves = [], [], []
        for number in (1, 2):
            a, b = runs((match.half == number) & ~seen)
            starts.extend(a.tolist())
            stops.extend(b.tolist())
            halves.extend([number] * len(a))
        start = np.asarray(starts, dtype=np.int64)
        stop = np.asarray(stops, dtype=np.int64)
        n = match.n_frames

        def at(frames):
            ok = (frames >= 0) & (frames < n)
            out = np.full(len(frames), np.nan)
            out[ok] = edge[frames[ok]]
            return out

        with np.errstate(all="ignore"):
            inside = np.array([np.nanmedian(speed[a:b]) for a, b in zip(start, stop, strict=True)])
        return {
            "start": start,
            "stop": stop,
            "half": np.asarray(halves, dtype=np.int8),
            "duration_s": (stop - start) / float(match.fps),
            "players_speed": inside,
            "edge_before_m": at(start - 1),
            "edge_after_m": at(stop),
        }

    return _memo(match, "ball_gaps", compute)


def nearest_by_team(match: Match) -> dict:
    """Nearest tracked player of each team to the ball, per frame.

    Keys: `distance` (n, 2) metres and `slot` (n, 2) track index, column 0 home and 1 away;
    NaN / -1 where the ball or that whole team is missing. Ties go to the lowest slot.
    """

    def compute():
        n = match.n_frames
        distance = np.full((n, 2), np.nan)
        slot = np.full((n, 2), -1, dtype=np.int64)
        for start in range(0, n, CHUNK):
            stop = min(n, start + CHUNK)
            delta = match.players[start:stop, :, :2] - match.ball[start:stop, None, :2]
            squared = delta[..., 0] ** 2 + delta[..., 1] ** 2
            squared = np.where(np.isnan(squared), np.inf, squared)
            for team in (0, 1):
                mine = np.where(match.team_of_track[None, :] == team, squared, np.inf)
                best = np.argmin(mine, axis=1)
                best_sq = mine[np.arange(stop - start), best]
                found = np.isfinite(best_sq)
                distance[start:stop, team][found] = np.sqrt(best_sq[found])
                slot[start:stop, team][found] = best[found]
        return {"distance": distance, "slot": slot}

    return _memo(match, "nearest_by_team", compute)


def kickoffs(
    match: Match,
    tolerance_m: float = 3.0,
    share: float = 0.9,
    min_players: int = 7,
    min_s: float = 1.0,
    merge_gap_s: float = 15.0,
    centre_m: float = 1.5,
    leave_m: float = 2.0,
    lag_s: float = 1.8,
) -> dict:
    """Kick-off set-ups: stretches where each team stands in its own half, and when the kick came.

    A set-up is a run of at least `min_s` in which at least `share` of each team's tracked players
    (at least `min_players` of them) are on their own side of the halfway line, within
    `tolerance_m`. Which side is a team's own is read per half from where its players are on
    average. Runs closer than `merge_gap_s` are one wait (a retaken kick-off, a player stepping
    over the line and back).

    Keys (arrays of equal length, one entry per set-up): `start`, `stop` (video frames, stop
    exclusive) and `kick`, the frame play restarts: the first frame the tracked ball is more than
    `leave_m` from the centre mark after having been within `centre_m` of it, or, when the ball
    was not seen on the mark, `lag_s` before the set-up ends (players need that long to cross
    the line; measured on the external matches' kick-off events).
    """

    def compute():
        n, fps = match.n_frames, match.fps
        length, width = match.pitch
        x = match.players[:, :, 0]
        tracked = ~np.isnan(x)
        setup = np.zeros(n, dtype=bool)
        for number in (1, 2):
            in_half = match.half == number
            if not in_half.any():
                continue
            with np.errstate(all="ignore"):
                centre_of = [
                    np.nanmedian(x[in_half][:, match.team_of_track == team]) for team in (0, 1)
                ]
            if np.isnan(centre_of).any():
                continue
            low_side = 0 if centre_of[0] < centre_of[1] else 1
            ok = in_half.copy()
            for team in (0, 1):
                mine = match.team_of_track == team
                count = tracked[:, mine].sum(axis=1)
                with np.errstate(invalid="ignore"):
                    if team == low_side:
                        own = x[:, mine] <= length / 2 + tolerance_m
                    else:
                        own = x[:, mine] >= length / 2 - tolerance_m
                inside = (own & tracked[:, mine]).sum(axis=1)
                ok &= (count >= min_players) & (inside >= share * count)
            setup |= ok

        starts, stops = runs(setup)
        keep = (stops - starts) >= min_s * fps
        starts, stops = starts[keep], stops[keep]
        merged: list[list[int]] = []
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            same_wait = (
                merged
                and start - merged[-1][1] <= merge_gap_s * fps
                and match.half[start] == match.half[merged[-1][0]]
            )
            if same_wait:
                merged[-1][1] = stop
            else:
                merged.append([start, stop])

        to_centre = np.hypot(match.ball[:, 0] - length / 2, match.ball[:, 1] - width / 2)
        kicks = []
        for start, stop in merged:
            kick = max(stop - int(round(lag_s * fps)), start)
            with np.errstate(invalid="ignore"):
                on_mark = np.flatnonzero(to_centre[start:stop] <= centre_m)
                if len(on_mark):
                    after = on_mark[-1] + 1
                    left = np.flatnonzero(to_centre[start + after : stop] > leave_m)
                    if len(left):
                        kick = start + after + int(left[0])
            kicks.append(kick)
        return {
            "start": np.array([m[0] for m in merged], dtype=np.int64),
            "stop": np.array([m[1] for m in merged], dtype=np.int64),
            "kick": np.array(kicks, dtype=np.int64),
        }

    key = (
        "kickoffs",
        tolerance_m,
        share,
        min_players,
        min_s,
        merge_gap_s,
        centre_m,
        leave_m,
        lag_s,
    )
    return _memo(match, key, compute)


def kickoff_wait(match: Match, wait_s: float = 300.0, wait_speed_m_s: float = 2.5) -> np.ndarray:
    """(n,) bool: frames where play is waiting for a kick-off.

    Every kick-off set-up (`kickoffs`) up to its kick. At the start of a half, the wait before the
    first set-up as well, if that set-up begins within `wait_s` and the players were not running
    before it (median `collective_speed` below `wait_speed_m_s`); a half whose data starts at the
    kick-off has no such wait.
    """

    def compute():
        fps = match.fps
        setups = kickoffs(match)
        speed = collective_speed(match)
        waiting = np.zeros(match.n_frames, dtype=bool)
        for start, kick in zip(setups["start"].tolist(), setups["kick"].tolist(), strict=True):
            waiting[start:kick] = True
        for number in (1, 2):
            in_half = np.flatnonzero(match.half == number)
            first = setups["start"][match.half[setups["start"]] == number][:1]
            if not len(in_half) or not len(first):
                continue
            begin, start = int(in_half[0]), int(first[0])
            before = speed[begin:start]
            if 0 < start - begin <= wait_s * fps and not np.isnan(before).all():
                if np.nanmedian(before) < wait_speed_m_s:
                    waiting[begin:start] = True
        return waiting

    return _memo(match, ("kickoff_wait", wait_s, wait_speed_m_s), compute)
