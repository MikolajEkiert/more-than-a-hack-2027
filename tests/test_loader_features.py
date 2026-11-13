import numpy as np

from possession import features
from possession.loader import parse_atd_raw
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
