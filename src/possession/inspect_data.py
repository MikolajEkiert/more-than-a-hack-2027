"""Aggregate-only data inspection: what the signals look like, with no frame-level output.

Every function returns a small plain dict of rates, percentiles, histograms and counts. Nothing
here returns or prints a frame index of an individual event, a per-frame value or a coordinate of
a specific frame, so the output is safe to show to an AI agent when the data guard is on. The size
of the result depends on the fixed set of statistics, never on the number of frames.

(Named inspect_data, not inspect, so it can never shadow the stdlib module.)

If a statistic you need is missing, add it here as an aggregate, with a test in
tests/test_inspect.py, instead of writing an ad-hoc script that loads the data.
"""

import numpy as np

import possession_metrics as pm
from possession import features
from possession.match import AWAY, HOME, Match
from possession.scoring import _runs

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
GAP_BUCKETS_S = (("<0.2s", 0.0, 0.2), ("0.2-1s", 0.2, 1.0), ("1-3s", 1.0, 3.0),
                 ("3-10s", 3.0, 10.0), (">10s", 10.0, np.inf))  # fmt: skip
NEAR_M = (1.0, 2.0, 5.0)
LONG_GAP_S = 3.0
SHORT_POSSESSION_S = (2.0, 5.0)


def _dist(values: np.ndarray) -> dict:
    """Percentiles (monotone non-decreasing) of a 1-D array, plus n, min and max."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    out = {"n": int(len(values)), "min": float(values.min())}
    for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES), strict=True):
        out[f"p{p:02d}"] = float(v)
    out["max"] = float(values.max())
    return out


def _rate(mask: np.ndarray, base: np.ndarray) -> float | None:
    n = int(base.sum())
    return float((mask & base).sum()) / n if n else None


def _half_runs(half: np.ndarray) -> list[tuple[int, int]]:
    """Graded spans, one list entry per contiguous stretch of each half."""
    return [r for number in (1, 2) for r in _runs(half == number)]


def _gap_stats(seen: np.ndarray, half: np.ndarray, fps: int) -> dict:
    lengths = [
        stop - start
        for number in (1, 2)
        for start, stop in _runs((half == number) & ~seen)
    ]  # fmt: skip
    seconds = np.asarray(lengths, dtype=np.float64) / fps
    hist = {name: int(((seconds >= lo) & (seconds < hi)).sum()) for name, lo, hi in GAP_BUCKETS_S}
    out = {"n_gaps": int(len(seconds)), "histogram": hist}
    if len(seconds):
        out.update(
            p50=float(np.percentile(seconds, 50)),
            p90=float(np.percentile(seconds, 90)),
            p99=float(np.percentile(seconds, 99)),
            max=float(seconds.max()),
        )
    return out


def _quartiles(values: np.ndarray) -> dict:
    """n and three quartiles, with the same keys whether or not there is anything to summarise."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "p25": None, "p50": None, "p75": None}
    q = np.percentile(values, [25, 50, 75])
    return {"n": int(len(values)), "p25": float(q[0]), "p50": float(q[1]), "p75": float(q[2])}


def _where(edge: np.ndarray) -> dict:
    known = edge[~np.isnan(edge)]
    if not len(known):
        return {"n": 0, "outside_share": None, "within_1m_share": None, "beyond_5m_share": None}
    return {
        "n": int(len(known)),
        "outside_share": float((known < 0).mean()),
        "within_1m_share": float(((known >= 0) & (known < 1)).mean()),
        "beyond_5m_share": float((known >= 5).mean()),
    }


def gaps(match: Match) -> dict:
    """The long ball gaps and what surrounds them: stoppages, or play with the ball lost?"""
    graded, seen = match.graded, match.ball_seen
    tracked = seen & graded
    found = features.ball_gaps(match)
    long = found["duration_s"] >= LONG_GAP_S
    speed = features.collective_speed(match)
    kin = features.ball_kinematics(match)
    return {
        "long_gaps": {
            "n": int(long.sum()),
            "frames_share_of_graded": float(
                (found["stop"][long] - found["start"][long]).sum() / max(int(graded.sum()), 1)
            ),
            "duration_s": _quartiles(found["duration_s"][long]),
            "players_speed_p90_inside_m_s": _quartiles(found["players_speed"][long]),
            "ball_last_seen": _where(found["edge_before_m"][long]),
            "ball_first_seen_after": _where(found["edge_after_m"][long]),
        },
        "players_speed_p90_m_s": {
            "ball_tracked": _quartiles(speed[tracked]),
            "ball_missing": _quartiles(speed[graded & ~seen]),
        },
        # The ATD has its own ball speed column; compare it with speed derived from positions.
        "ball_speed_m_s": {
            "atd_column": _quartiles(match.ball[tracked, 2]),
            "from_positions": _quartiles(kin["speed"][tracked]),
        },
    }


def ball(match: Match) -> dict:
    graded, seen = match.graded, match.ball_seen
    length, width = match.pitch
    tracked = seen & graded
    x, y = match.ball[tracked, 0], match.ball[tracked, 1]
    outside = (x < 0) | (x > length) | (y < 0) | (y > width)
    return {
        "graded_frames": int(graded.sum()),
        "tracked_share": {
            "overall": _rate(seen, graded),
            "half1": _rate(seen, match.half == 1),
            "half2": _rate(seen, match.half == 2),
        },
        "missing_gap_s": _gap_stats(seen, match.half, match.fps),
        "speed_m_s": _dist(match.ball[tracked, 2]),
        "outside_pitch_share": float(outside.mean()) if len(outside) else None,
    }


def players(match: Match) -> dict:
    graded = match.graded
    present = ~np.isnan(match.players[graded, :, 0])
    per_frame = present.sum(axis=1)
    team = match.team_of_track
    distance, _ = features.nearest_player(match)
    near = distance[graded]
    near = near[np.isfinite(near)]
    return {
        "graded_frames": int(graded.sum()),
        "tracked_per_frame": {
            **_dist(per_frame),
            "share_below_20": float((per_frame < 20).mean()) if len(per_frame) else None,
            "share_above_22": float((per_frame > 22).mean()) if len(per_frame) else None,
        },
        "mean_tracked_per_team": {
            "home": float(present[:, team == HOME].sum(axis=1).mean()) if len(present) else None,
            "away": float(present[:, team == AWAY].sum(axis=1).mean()) if len(present) else None,
        },
        "speed_m_s": _dist(match.players[graded, :, 2][present]),
        "nearest_to_ball_m": {
            **_dist(near),
            **{f"share_within_{int(m)}m": float((near <= m).mean()) if len(near) else None
               for m in NEAR_M},
        },
    }  # fmt: skip


def _tag(item: dict, key: str):
    for k, v in item["tags"].items():
        if k.lower() == key.lower():
            return v or "N/A"
    return "N/A"


def tagging(match: Match) -> dict:
    if match.tagging is None:
        return {"available": False}
    graded, seen = match.graded, match.ball_seen
    items = match.tagging

    def count(values):
        out: dict = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    any_tag = np.zeros(match.n_frames, dtype=bool)
    for item in items:
        any_tag[max(0, item["start_frame"]) : max(0, item["end_frame"])] = True
    sp = features.tagging_mask(match, "SET PIECES")
    sp_items = [i for i in items if i["code"].lower() == "set pieces"]
    return {
        "available": True,
        "n_instances": len(items),
        "per_code": count(i["code"] for i in items),
        "per_team_tag": count(_tag(i, "Team") for i in items),
        "set_piece_types": count(_tag(i, "Type") for i in sp_items),
        "graded_share_covered": {
            "any_tag": _rate(any_tag, graded),
            "set_pieces": _rate(sp, graded),
        },
        "ball_tracked_share": {
            "inside_set_pieces": _rate(seen, graded & sp),
            "outside_set_pieces": _rate(seen, graded & ~sp),
        },
    }


def kickoffs(match: Match) -> dict:
    """The formation-based kick-off rule next to the kick-offs in Smart Tagging (aggregates only).

    The tags are an independent source for the same events (two starts of a half and one after
    each goal). A set-up the rule finds with no tagged kick-off near it would be a false stoppage
    inside play; a tagged kick-off with no set-up near it is one the rule misses.
    """
    fps, graded = match.fps, match.graded
    found = features.kickoffs(match)
    starts, kicks = found["start"].tolist(), found["kick"].tolist()
    out = {
        "setups_found": len(kicks),
        "setup_until_kick_s": sorted(
            round((k - s) / fps, 1) for s, k in zip(starts, kicks, strict=True)
        ),
        "graded_share_waiting_for_kickoff": _rate(features.kickoff_wait(match), graded),
    }
    if match.tagging is None:
        return {**out, "tagging_available": False}
    tagged = [i for i in match.tagging if str(_tag(i, "Type")).lower().startswith("kick off")]
    windows = [(i["start_frame"], i["end_frame"]) for i in tagged]
    near = 10 * fps

    def gap(frame, window):  # frames between `frame` and the window, 0 inside it
        a, b = window
        return 0 if a <= frame <= b else min(abs(frame - a), abs(frame - b))

    return {
        **out,
        "tagging_available": True,
        "tagged_kickoffs": len(tagged),
        "tagged_kickoff_types": dict(sorted(_count(_tag(i, "Type") for i in tagged).items())),
        "tag_window_s": sorted(round((b - a) / fps, 1) for a, b in windows),
        "tagged_with_a_found_kick_within_10s": sum(
            any(gap(k, w) <= near for k in kicks) for w in windows
        ),
        "found_with_a_tag_within_10s": sum(any(gap(k, w) <= near for w in windows) for k in kicks),
        "found_kick_after_tag_start_s": sorted(
            round((k - w[0]) / fps, 1) for w in windows for k in kicks if gap(k, w) <= near
        ),
    }


def _count(values) -> dict:
    out: dict = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


STOPPAGE_BUCKETS_S = (("<3s", 0.0, 3.0), ("3-10s", 3.0, 10.0), ("10-30s", 10.0, 30.0),
                      (">30s", 30.0, np.inf))  # fmt: skip
ON_TIME_S = 0.5  # a predicted stoppage edge this close to the true one counts as on time
MOVING_M_S = 4.0  # a tracked ball this fast inside the pitch looks like play, not a stoppage
# Players "running": the 90th percentile of their speeds, averaged over 1 s, at or above 2.5 m/s,
# halfway between the stoppage and play means of `ball_state.stoppage_evidence` (1.6 and 3.5 m/s).
# Independent of the ball, so it can say how much of a set of frames is play.
RUNNING_M_S = 2.5
RUNNING_WINDOW_S = 1.0


def _running(match: Match) -> np.ndarray:
    speed = features.collective_speed(match)
    window = max(int(round(RUNNING_WINDOW_S * match.fps)), 1)
    ok = np.isfinite(speed)
    kernel = np.ones(window)
    total = np.convolve(np.where(ok, speed, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(np.float64), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count) >= RUNNING_M_S


def _graded_runs(mask: np.ndarray, half: np.ndarray) -> list[tuple[int, int]]:
    return [r for number in (1, 2) for r in _runs(mask & (half == number))]


def deadball_parts(labels, truth, match: Match) -> dict:
    """Per-stoppage quantities behind `deadball_errors`: lengths and edge lags in seconds and
    boolean outcomes, so that a caller can pool them over several matches. No frame indices."""
    labels, truth = np.asarray(labels).astype("<U1"), np.asarray(truth).astype("<U1")
    fps, half, seen = match.fps, match.half, match.ball_seen
    graded = (half > 0) & (truth != "X")
    pred, true = (labels == "D") & graded, (truth == "D") & graded
    true_runs, pred_runs = _graded_runs(true, half), _graded_runs(pred, half)
    pred_starts = np.array([a for a, _ in pred_runs], dtype=np.int64)
    pred_stops = np.array([b for _, b in pred_runs], dtype=np.int64)
    length, found, start_lag, end_lag = [], [], [], []
    for a, b in true_runs:
        length.append((b - a) / fps)
        hit = np.flatnonzero((pred_starts < b) & (pred_stops > a))
        found.append(bool(len(hit)))
        if len(hit):
            start_lag.append((pred_starts[hit[0]] - a) / fps)
            end_lag.append((pred_stops[hit[-1]] - b) / fps)
    false_tracked = [bool(seen[a:b].mean() > 0.5) for a, b in pred_runs if not true[a:b].any()]
    return {
        "frames": {
            "graded": int(graded.sum()),
            "true_dead": int(true.sum()),
            "pred_dead": int(pred.sum()),
            "missed_dead": int((true & ~pred).sum()),
            "false_dead": int((pred & ~true).sum()),
            "false_dead_ball_tracked": int((pred & ~true & seen).sum()),
            "true_dead_ball_missing": int((true & ~seen).sum()),
            "true_live": int((graded & ~true).sum()),
            "true_live_ball_missing": int((graded & ~true & ~seen).sum()),
        },
        "length_s": length,
        "found": found,
        "start_lag_s": start_lag,
        "end_lag_s": end_lag,
        "false_stoppage_ball_tracked": false_tracked,
    }


def _share(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def summarise_deadball(parts: list[dict]) -> dict:
    """Aggregates of one or more `deadball_parts` results, pooled."""
    frames = {k: sum(p["frames"][k] for p in parts) for k in parts[0]["frames"]}
    length = np.concatenate([np.asarray(p["length_s"], float) for p in parts])
    found = np.concatenate([np.asarray(p["found"], bool) for p in parts])
    start = np.concatenate([np.asarray(p["start_lag_s"], float) for p in parts])
    end = np.concatenate([np.asarray(p["end_lag_s"], float) for p in parts])
    tracked = np.concatenate([np.asarray(p["false_stoppage_ball_tracked"], bool) for p in parts])
    by_length = {}
    for name, lo, hi in STOPPAGE_BUCKETS_S:
        inside = (length >= lo) & (length < hi)
        by_length[name] = {
            "n": int(inside.sum()),
            "missed_share": _share(int((~found & inside).sum()), int(inside.sum())),
        }
    return {
        "true_dead_share": _share(frames["true_dead"], frames["graded"]),
        "pred_dead_share": _share(frames["pred_dead"], frames["graded"]),
        "missed_share_of_true_dead_frames": _share(frames["missed_dead"], frames["true_dead"]),
        "false_share_of_true_live_frames": _share(frames["false_dead"], frames["true_live"]),
        "false_dead_frames_ball_tracked_share": _share(
            frames["false_dead_ball_tracked"], frames["false_dead"]
        ),
        "ball_missing_given_dead": _share(frames["true_dead_ball_missing"], frames["true_dead"]),
        "ball_missing_given_live": _share(frames["true_live_ball_missing"], frames["true_live"]),
        "true_stoppages": int(len(length)),
        "missed_stoppages_share": _share(int((~found).sum()), int(len(found))),
        "missed_by_length": by_length,
        "start_lag_s": _dist(start),
        "start_on_time_share": _share(int((np.abs(start) <= ON_TIME_S).sum()), int(len(start))),
        "end_lag_s": _dist(end),
        "end_on_time_share": _share(int((np.abs(end) <= ON_TIME_S).sum()), int(len(end))),
        "false_stoppages": int(len(tracked)),
        "false_stoppages_ball_tracked_share": _share(int(tracked.sum()), int(len(tracked))),
    }


def deadball_errors(labels, truth, match: Match) -> dict:
    """How predicted stoppages (D) compare with the true ones: missed and false stoppages, edge
    lags, and whether the ball was tracked where a stoppage was invented. Aggregates only."""
    return summarise_deadball([deadball_parts(labels, truth, match)])


def regime(match: Match, labels, truth=None) -> dict:
    """Which ball regime the tracking is in, read from the stoppages of `labels`.

    Of the frames labelled D, how many have a tracked ball, and of those how many have it moving
    at MOVING_M_S or more inside the pitch (that looks like play, i.e. a false stoppage). With a
    truth, the same share for the true and for the false stoppages, and P(ball missing | dead or
    live): the reference points for reading the numbers of a match that has no truth.
    """
    labels = np.asarray(labels).astype("<U1")
    speed = features.ball_kinematics(match)["speed"]
    with np.errstate(invalid="ignore"):
        moving = (speed >= MOVING_M_S) & (features.ball_edge_distance(match) >= 0)
    graded, seen = match.graded, match.ball_seen
    running = _running(match)
    pred = (labels == "D") & graded
    out = {
        "pred_dead_share": _rate(pred, graded),
        "pred_dead_ball_tracked_share": _rate(seen, pred),
        "pred_dead_tracked_moving_share": _rate(moving, pred & seen),
        "ball_missing_share": _rate(~seen, graded),
        "running_given_ball_missing": _rate(running, graded & ~seen),
        "running_given_pred_dead_tracked": _rate(running, pred & seen),
        "running_given_pred_dead_missing": _rate(running, pred & ~seen),
        "running_given_pred_live": _rate(running, graded & np.isin(labels, ["H", "A"])),
    }
    if truth is None:
        return out
    truth = np.asarray(truth).astype("<U1")
    true = (truth == "D") & graded
    live = graded & np.isin(truth, ["H", "A"])
    return {
        **out,
        "true_dead_ball_tracked_share": _rate(seen, true),
        "true_dead_tracked_moving_share": _rate(moving, true & seen),
        "false_dead_tracked_moving_share": _rate(moving, pred & live & seen),
        "ball_missing_given_dead": _rate(~seen, true),
        "ball_missing_given_live": _rate(~seen, live),
        "running_given_true_dead": _rate(running, true),
        "running_given_true_live": _rate(running, live),
        "running_given_true_dead_tracked": _rate(running, true & seen),
        "running_given_true_live_tracked": _rate(running, live & seen),
    }


def label_stats(labels, half: np.ndarray, fps: int) -> dict:
    """Aggregates of a submission's labels over the graded frames given by `half` (1, 2, 0)."""
    codes = pm.label_codes(labels)  # a label array or the DataFrame from read_label_csv
    graded = half > 0
    runs = _half_runs(half)
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    seconds = np.array([(b - a) / fps for a, b, _ in spells], dtype=np.float64)
    n_graded = int(graded.sum())
    minutes = n_graded / fps / 60
    n_trans = len(pm.transitions(filled, runs))

    def shares(mask):
        n = int(mask.sum())
        return {
            name: (float((codes[mask] == i).sum()) / n if n else None)
            for i, name in enumerate(pm.LABELS)
        }

    out = {
        "graded_frames": n_graded,
        "label_share": {"graded": shares(graded), "half1": shares(half == 1),
                        "half2": shares(half == 2)},
        "x_share_graded": float((codes[graded] == pm.X).mean()) if n_graded else None,
        "n_possessions": len(spells),
        "possession_s": _dist(seconds),
        "n_shorter_than": {f"{s:g}s": int((seconds < s).sum()) for s in SHORT_POSSESSION_S},
        "transitions_per_min": n_trans / minutes if minutes else None,
    }  # fmt: skip
    return out


def labels(csv_path: str, game: str) -> dict:
    """label_stats for a submission CSV. Needs only the CSV and the game's half boundaries."""
    from possession.loader import half_mask

    frame = pm.read_label_csv(csv_path)
    return label_stats(frame, half_mask(game, len(frame)), _fps(game))


def _fps(game: str) -> int:
    import games

    return int(games.GAMES[game]["fps"])


# -- text rendering ----------------------------------------------------------------------------
def flatten(data: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(flatten(value, name + "."))
        elif isinstance(value, float):
            rows.append((name, f"{value:.4g}"))
        else:
            rows.append((name, "-" if value is None else str(value)))
    return rows


def render(data: dict) -> str:
    rows = flatten(data)
    width = max((len(k) for k, _ in rows), default=0)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)
