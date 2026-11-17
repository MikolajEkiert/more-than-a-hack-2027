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
    for topic in ("ball", "players", "gaps"):
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


def test_cli_inspect_regime_json(monkeypatch, capsys, match):
    monkeypatch.setattr("possession.loader.load_match", lambda game, verbose=True: match)
    assert cli.main(["inspect", "regime", "--game", "SYNTH", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {"pred_dead_share", "running_given_pred_dead_tracked"} <= set(out)
    for name, value in _rates(out):
        assert 0 <= value <= 1, name


def test_gap_aggregates(match):
    out = inspect_data.gaps(match)
    assert out["long_gaps"]["n"] >= 0
    for name, value in _rates(out):
        assert 0 <= value <= 1, name
    # make_match fills the speed column with 5.0 while the ball follows a random walk: the two
    # sources are reported side by side so a disagreement like the real data's is visible
    assert out["ball_speed_m_s"]["atd_column"]["p50"] == pytest.approx(5.0)
    assert out["ball_speed_m_s"]["from_positions"]["n"] > 0


def test_kickoff_audit_counts_setups_and_tags():
    from dataclasses import replace

    from possession import features, inspect_data

    match = make_match(n=2000, ungraded_head=0)
    assert inspect_data.kickoffs(match)["tagging_available"] is False

    found = {
        "start": np.array([100, 900]),
        "stop": np.array([200, 1000]),
        "kick": np.array([180, 990]),
    }
    tags = [
        {
            "code": "SET PIECES",
            "start_frame": 100,
            "end_frame": 225,
            "tags": {"Type": "Kick Off (Start)"},
        },
        {
            "code": "SET PIECES",
            "start_frame": 1500,
            "end_frame": 1625,
            "tags": {"Type": "Kick Off (After Goal)"},
        },
        {"code": "SET PIECES", "start_frame": 400, "end_frame": 525, "tags": {"Type": "Throw-in"}},
    ]
    tagged = replace(match, tagging=tags, _cache={})
    key = ("kickoffs", 3.0, 0.9, 7, 1.0, 15.0, 1.5, 2.0, 1.8)  # features.kickoffs' defaults
    tagged._cache[key] = found
    assert features.kickoffs(tagged) is found
    out = inspect_data.kickoffs(tagged)
    assert out["setups_found"] == 2 and out["tagged_kickoffs"] == 2  # the throw-in is not one
    assert out["tagged_with_a_found_kick_within_10s"] == 1  # the second tag has no set-up near it
    assert out["found_with_a_tag_within_10s"] == 1  # the second set-up has no tag near it
    assert out["found_kick_after_tag_start_s"] == [3.2]  # frame 180 against a window from 100
    assert out["setup_until_kick_s"] == [3.2, 3.6]


def _stoppage_case():
    m = make_match(n=1000, ball_missing=0.0)
    truth = np.full(m.n_frames, "H", dtype="<U1")
    truth[~m.graded] = "X"
    truth[300:400] = "D"  # 4 s, found 1 s late, ends on time
    truth[600:650] = "D"  # 2 s, missed
    labels = truth.copy()
    labels[300:325] = "H"
    labels[600:650] = "H"
    labels[800:850] = "D"  # invented, with the ball tracked
    return m, labels, truth


def test_deadball_errors_counts_missed_false_and_late():
    m, labels, truth = _stoppage_case()
    out = inspect_data.deadball_errors(labels, truth, m)
    assert out["true_stoppages"] == 2 and out["missed_stoppages_share"] == 0.5
    assert out["start_lag_s"]["max"] == pytest.approx(1.0)
    assert out["start_on_time_share"] == 0.0 and out["end_on_time_share"] == 1.0
    assert out["false_stoppages"] == 1 and out["false_stoppages_ball_tracked_share"] == 1.0
    assert out["missed_by_length"]["<3s"] == {"n": 1, "missed_share": 1.0}
    assert out["missed_by_length"]["3-10s"] == {"n": 1, "missed_share": 0.0}
    assert out["missed_share_of_true_dead_frames"] == pytest.approx(75 / 150, abs=1e-4)
    assert out["ball_missing_given_dead"] == 0.0


def test_deadball_parts_pool_like_one_match():
    m, labels, truth = _stoppage_case()
    part = inspect_data.deadball_parts(labels, truth, m)
    pooled = inspect_data.summarise_deadball([part, part])
    single = inspect_data.summarise_deadball([part])
    assert pooled["true_stoppages"] == 2 * single["true_stoppages"]
    assert pooled["missed_stoppages_share"] == single["missed_stoppages_share"]


def test_regime_with_and_without_truth():
    m, labels, truth = _stoppage_case()
    out = inspect_data.regime(m, labels)
    assert out["pred_dead_ball_tracked_share"] == 1.0
    for name, value in _rates(out):
        assert 0 <= value <= 1, name
    full = inspect_data.regime(m, labels, truth=truth)
    assert full["ball_missing_given_dead"] == 0.0 and full["ball_missing_given_live"] == 0.0
    assert full["false_dead_tracked_moving_share"] is not None
    assert inspect_data.regime(m, truth, truth=truth)["false_dead_tracked_moving_share"] is None


def _regime_case():
    """3000 frames, one half: play, a stoppage of 8 s with the ball lost but placed (still) for
    its last 2 s, play without a ball for 5 s running into a 4 s stoppage, play to the end."""
    m = make_match(n=3000, ball_missing=0.0, ungraded_head=0)
    labels = np.full(m.n_frames, "H", dtype="<U1")
    labels[1000:1200] = "D"
    labels[2000:2100] = "D"
    m.ball[1000:1150] = np.nan
    m.ball[1150:1200, :2] = m.ball[1200, :2]  # lying still at the restart spot
    m.ball[1875:2100] = np.nan  # lost 5 s before the second stoppage, and during it
    m.ball[2500:2510] = np.nan  # a dropout of 0.4 s
    return m, labels


def test_regime_vector_reads_ball_loss_and_placed_balls():
    m, labels = _regime_case()
    out = inspect_data.regime_vector(m, labels)
    assert out["missing_given_dead"] == pytest.approx(250 / 300, abs=1e-4)
    assert out["missing_given_live"] == pytest.approx(135 / 2700, abs=1e-4)
    assert out["live_missing_lead_in"] == pytest.approx(125 / 2700, abs=1e-4)
    assert out["live_missing_free"] == 0.0 and out["live_missing_lead_out"] == 0.0
    assert out["live_dropouts_per_10min_live"] == pytest.approx(1 / (2700 / 25 / 60) * 10, 1e-3)
    assert out["lead_in_s_p50"] == pytest.approx(5.0)
    assert out["stoppage_end_tracked"] == 0.5 and out["stoppage_start_tracked"] == 0.0
    assert out["stoppage_end_still"] == 0.5  # the placed ball, not the lost one
    assert out["end_visible_s_p50"] == pytest.approx(2.0)
    assert (
        out["dead_tracked_first_third"] == 0.0 and out["dead_tracked_last_third"] == 0.5
    )  # 50 of 67 + 33
    assert out["dead_tracked_still"] > 0.9
    assert out["stoppages_per_10min"] == pytest.approx(2 / (3000 / 25 / 60) * 10, 1e-3)
    assert out["stoppage_s_p50"] == pytest.approx(6.0)  # 8 s and 4 s
    assert out["dead_share"] == pytest.approx(0.1, abs=1e-4)
    for name, value in out.items():  # shares stay shares; nothing is a per-frame array
        assert value is None or np.isscalar(value), name


def test_regime_vector_without_labels_is_the_tracking_part():
    m, labels = _regime_case()
    alone, both = inspect_data.regime_vector(m), inspect_data.regime_vector(m, labels)
    assert set(alone) < set(both) and all(both[k] == v for k, v in alone.items())
    assert alone["ball_missing"] == pytest.approx(385 / 3000, abs=1e-4)
    assert alone["gaps_under_1s_per_10min"] == pytest.approx(1 / 2 * 10, 1e-3)
    assert alone["gaps_over_3s_per_10min"] == pytest.approx(2 / 2 * 10, 1e-3)
    assert alone["still_runs_per_10min"] >= 5.0  # the placed ball (2 s) is one still run
    small = inspect_data.regime_vector(make_match(n=600, ball_missing=0.0))
    assert len(small) == len(alone)  # the size never depends on the number of frames
