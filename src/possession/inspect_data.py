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
