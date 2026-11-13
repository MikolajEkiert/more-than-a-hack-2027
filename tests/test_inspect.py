import json

import numpy as np
import pytest

from possession import cli, inspect_data
from possession.match import Match

from .conftest import make_match


def _monotone(dist: dict) -> bool:
    keys = ["min"] + [k for k in dist if k.startswith("p")] + ["max"]
    values = [dist[k] for k in keys]
    return all(a <= b for a, b in zip(values, values[1:], strict=False))


def _rates(data, path=""):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _rates(value, path + key + ".")
        elif "share" in key and value is not None:
            yield path + key, value


def test_ball_aggregates(match):
    out = inspect_data.ball(match)
    assert {"tracked_share", "missing_gap_s", "speed_m_s", "outside_pitch_share"} <= set(out)
    for name, value in _rates(out):
        assert 0 <= value <= 1, name
    assert _monotone(out["speed_m_s"])
    gaps = out["missing_gap_s"]
    assert sum(gaps["histogram"].values()) == gaps["n_gaps"] > 0
    assert gaps["p50"] <= gaps["p90"] <= gaps["p99"] <= gaps["max"]


def test_ball_outside_pitch_is_counted():
    m = make_match(n=500, ball_missing=0.0)
    m.ball[300, 0] = m.pitch[0] + 5
    m.ball[301, 1] = -1
    assert inspect_data.ball(m)["outside_pitch_share"] == pytest.approx(2 / 300)


def test_players_aggregates(match):
    out = inspect_data.players(match)
    for name, value in _rates(out):
        assert 0 <= value <= 1, name
    assert _monotone(out["tracked_per_frame"]) and _monotone(out["speed_m_s"])
    assert _monotone(out["nearest_to_ball_m"])
    near = out["nearest_to_ball_m"]
    assert near["share_within_1m"] <= near["share_within_2m"] <= near["share_within_5m"]
    assert out["tracked_per_frame"]["share_below_20"] == 1.0  # synthetic match has 8 slots
    assert out["mean_tracked_per_team"]["home"] <= 4


def test_tagging_absent_is_reported(match):
    assert inspect_data.tagging(match) == {"available": False}


def test_tagging_aggregates():
    m = make_match(n=1000, ball_missing=0.0)
    items = [
        {"code": "SET PIECES", "start_frame": 300, "end_frame": 400, "tags": {"Team": "Home",
         "Type": "Throw-in"}},
        {"code": "SET PIECES", "start_frame": 500, "end_frame": 550, "tags": {"Team": None,
         "Type": "Corner"}},
        {"code": "SHOTS", "start_frame": 600, "end_frame": 620, "tags": {"Team": "Away"}},
    ]  # fmt: skip
    m2 = Match(**{**m.__dict__, "tagging": items, "_cache": {}})
    out = inspect_data.tagging(m2)
    assert out["per_code"] == {"SET PIECES": 2, "SHOTS": 1}
    assert out["per_team_tag"] == {"Away": 1, "Home": 1, "N/A": 1}
    assert out["set_piece_types"] == {"Corner": 1, "Throw-in": 1}
    assert out["graded_share_covered"]["set_pieces"] == pytest.approx(150 / 800)
    assert out["graded_share_covered"]["any_tag"] == pytest.approx(170 / 800)
    assert out["ball_tracked_share"]["inside_set_pieces"] == 1.0


def labels_example():
    """Graded frames 2..10 of 12, one half; labels outside it are junk that must be ignored.

    graded: H H X H A A D D H  ->  X filled to H: H H H H A A D D H
    possessions (H or A runs): HHHH (4), AA (2), H (1)  => 3, lasting 4, 2, 1 frames.
    """
    seq = list("DD" + "HHXHAADDH" + "A")
    half = np.array([0, 0] + [1] * 9 + [0], dtype=np.int8)
    return np.array(seq, dtype=object), half


def test_label_stats_hand_example():
    labels, half = labels_example()
    out = inspect_data.label_stats(labels, half, fps=1)
    assert out["graded_frames"] == 9
    assert out["n_possessions"] == 3
    assert out["possession_s"]["max"] == 4 and out["possession_s"]["min"] == 1
    assert out["n_shorter_than"] == {"2s": 1, "5s": 3}
    assert out["x_share_graded"] == pytest.approx(1 / 9)
    assert out["label_share"]["graded"]["H"] == pytest.approx(4 / 9)
    assert out["label_share"]["graded"]["D"] == pytest.approx(2 / 9)
    assert out["label_share"]["half2"]["H"] is None
    # changes: H->A, A->D, D->H over 9 s = 0.15 min
    assert out["transitions_per_min"] == pytest.approx(3 / (9 / 60))


def test_labels_from_csv(tmp_path, monkeypatch):
    labels, half = labels_example()
    path = tmp_path / "x.csv"
    path.write_text("frame,label\n" + "\n".join(f"{i},{v}" for i, v in enumerate(labels)) + "\n")
    monkeypatch.setattr("possession.loader.half_mask", lambda game, n: half)
    monkeypatch.setattr(inspect_data, "_fps", lambda game: 1)
    assert inspect_data.labels(str(path), "ANY")["n_possessions"] == 3


def test_output_size_independent_of_n_frames():
    small, large = (make_match(n=n) for n in (3000, 30000))
    for topic in ("ball", "players"):
        a = getattr(inspect_data, topic)(small)
        b = getattr(inspect_data, topic)(large)
        assert [k for k, _ in inspect_data.flatten(a)] == [k for k, _ in inspect_data.flatten(b)]
        assert len(inspect_data.render(b).splitlines()) < 40
        assert len(inspect_data.render(b)) < 3000
    rng = np.random.default_rng(0)
    for n in (3000, 30000):
        labels = rng.choice(list("HADX"), size=n).astype(object)
        out = inspect_data.label_stats(labels, np.ones(n, dtype=np.int8), 25)
        assert len(inspect_data.render(out).splitlines()) < 40


def test_cli_inspect_json(monkeypatch, capsys, match):
    monkeypatch.setattr("possession.loader.load_match", lambda game, verbose=True: match)
    assert cli.main(["inspect", "ball", "--game", "SYNTH", "--json"]) == 0
    assert "tracked_share" in json.loads(capsys.readouterr().out)
    assert cli.main(["inspect", "players", "--game", "SYNTH"]) == 0
    assert "tracked_per_frame.p50" in capsys.readouterr().out
