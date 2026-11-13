import numpy as np

from possession import features
from possession.loader import parse_atd_raw
from possession.match import AWAY, HOME, Match
from tests.conftest import make_match

PITCH = (100.0, 50.0)


def test_parse_atd_raw(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_text(
        "1:0.1,0.2,1.0;NaN,NaN,NaN:0.5,0.5,9.0\n"
        "2:0.3,0.4,2.0;0.6,0.8,3.0:NaN,NaN,NaN\n"
        "garbage line\n"
    )
    ball, players = parse_atd_raw(str(raw), n_frames=3, n_slots=2, pitch=PITCH)

    # frame_id 1 is video frame 0; positions are scaled to metres, speed is not
    np.testing.assert_allclose(ball[0], [50.0, 25.0, 9.0])
    np.testing.assert_allclose(players[0, 0], [10.0, 10.0, 1.0])
    assert np.isnan(players[0, 1]).all()
    assert np.isnan(ball[1]).all()
    np.testing.assert_allclose(players[1, 1], [60.0, 40.0, 3.0])
    assert np.isnan(ball[2]).all() and np.isnan(players[2]).all()  # never seen -> NaN


def test_empty_fields_are_missing(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_text("1:0.1,0.2,1.0;,,:,,\n")
    ball, players = parse_atd_raw(str(raw), n_frames=1, n_slots=2, pitch=PITCH)
    np.testing.assert_allclose(players[0, 0], [10.0, 10.0, 1.0])
    assert np.isnan(players[0, 1]).all() and np.isnan(ball[0]).all()


def test_nearest_player_matches_brute_force():
    match = make_match(n=500, seed=3)
    distance, team = features.nearest_player(match)
    for frame in range(match.n_frames):
        b = match.ball[frame, :2]
        d = np.hypot(*(match.players[frame, :, :2] - b).T)
        if np.isnan(b).any() or np.isnan(d).all():
            assert team[frame] == -1 and np.isnan(distance[frame])
            continue
        best = int(np.nanargmin(d))
        assert team[frame] == match.team_of_track[best]
        assert np.isclose(distance[frame], d[best])


def test_features_are_memoised():
    match = make_match(n=200)
    assert features.nearest_player(match) is features.nearest_player(match)


def _moving_ball_match(n=200, fps=25, speed=10.0):
    """One player and a ball on a straight line at constant speed, with a gap in the middle."""
    t = np.arange(n) / fps
    ball = np.stack([5.0 + speed * t, np.full(n, 20.0), np.full(n, 99.0)], axis=1)
    ball[80:120] = np.nan
    players = np.zeros((n, 2, 3))
    players[:, 0] = [10.0, 10.0, 1.0]
    players[:, 1] = [20.0, 20.0, 6.0]
    return Match(
        game="SYNTH", fps=fps, pitch=(105.0, 68.0), ball=ball, players=players,
        team_of_track=np.array([HOME, AWAY], dtype=np.int8), half=np.ones(n, dtype=np.int8),
    )  # fmt: skip


def test_ball_kinematics_come_from_positions_not_the_speed_column():
    match = _moving_ball_match(speed=10.0)
    kin = features.ball_kinematics(match)
    seen = match.ball_seen
    # the ATD speed column says 99; the positions say 10 m/s, on every tracked frame
    np.testing.assert_allclose(kin["speed"][seen], 10.0, atol=1e-6)
    np.testing.assert_allclose(kin["xy"][seen], match.ball[seen, :2], atol=1e-6)
    assert np.isnan(kin["speed"][~seen]).all() and np.isnan(kin["xy"][~seen]).all()
    np.testing.assert_allclose(kin["turn_deg"][10:70], 0.0, atol=1e-4)
    np.testing.assert_allclose(kin["accel"][10:70], 0.0, atol=1e-6)


def test_ball_kinematics_see_a_right_angle_turn():
    match = _moving_ball_match()
    ball = match.ball.copy()
    ball[:40, 0], ball[:40, 1] = 5.0 + np.arange(40) * 0.4, 20.0
    ball[40:80, 0], ball[40:80, 1] = ball[39, 0], 20.0 + np.arange(1, 41) * 0.4
    turned = Match(
        game="SYNTH", fps=25, pitch=match.pitch, ball=ball, players=match.players,
        team_of_track=match.team_of_track, half=match.half,
    )  # fmt: skip
    turn = features.ball_kinematics(turned)["turn_deg"]
    assert 80 < np.nanmax(turn[30:50]) <= 100
    assert np.nanmax(turn[5:25]) < 1


def test_collective_speed_and_ball_gaps():
    match = _moving_ball_match()
    speed = features.collective_speed(match)
    np.testing.assert_allclose(speed, 5.5)  # 90th percentile of [1, 6]
    gaps = features.ball_gaps(match)
    assert gaps["start"].tolist() == [80] and gaps["stop"].tolist() == [120]
    np.testing.assert_allclose(gaps["duration_s"], [1.6])
    np.testing.assert_allclose(gaps["players_speed"], [5.5])
    np.testing.assert_allclose(gaps["edge_before_m"], [20.0])  # y = 20 is the nearest line
    assert features.ball_edge_distance(match)[0] == 5.0


def test_runs_helper():
    starts, stops = features.runs(np.array([0, 1, 1, 0, 1], dtype=bool))
    assert starts.tolist() == [1, 4] and stops.tolist() == [3, 5]
