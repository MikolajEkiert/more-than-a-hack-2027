"""SkillCorner open data as package containers (possession/skillcorner.py), on synthetic files."""

import json

import numpy as np

from possession import skillcorner as sc

H, A, D, X = 0, 1, 2, 3


def _write(tmp_path, n_frames=40, untracked=(), undetected=()):
    meta = {
        "id": 1,
        "home_team": {"id": 10},
        "away_team": {"id": 20},
        "pitch_length": 104,
        "pitch_width": 68,
        "players": [{"id": 100 + k, "team_id": 10 if k < 4 else 20} for k in range(8)],
    }
    (tmp_path / "m.json").write_text(json.dumps(meta))
    lines = [json.dumps({"frame": 0, "period": None, "ball_data": {}, "possession": {}})]
    for f in range(1, n_frames + 1):
        period = 1 if f <= n_frames // 2 else 2
        group = "home team" if f < 10 else (None if f < 15 else "away team")
        players = [] if f in untracked else [
            {"player_id": 100 + k, "x": float(k), "y": -float(k), "is_detected": True}
            for k in range(8)
        ]  # fmt: skip
        lines.append(
            json.dumps(
                {
                    "frame": f,
                    "period": period,
                    "ball_data": {"x": 1.0, "y": 2.0, "is_detected": f not in undetected},
                    "possession": {"group": group},
                    "player_data": players,
                }
            )
        )
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path / "t.jsonl", tmp_path / "m.json"


def test_parse_gives_a_25_hz_container_with_provider_labels(tmp_path):
    m = sc.parse(*_write(tmp_path, untracked={6}, undetected={3}))
    assert m["name"] == "SC-1" and m["team"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    play = m["half"] > 0
    first = m["half"] == 1
    # 20 frames at 10 Hz (frames 1-20) span 1.9 s: 48 frames at 25 Hz
    assert first.sum() == 48
    # centred metres moved into [0, L] x [0, W]; player k stands at (k, -k) from the centre
    np.testing.assert_allclose(np.nanmax(m["px"][first][:, 0]), 52.0)
    np.testing.assert_allclose(np.nanmax(m["py"][first][:, 3]), 31.0)
    np.testing.assert_allclose(np.nanmax(m["bx"]), 53.0)
    labels = set(m["label"][play].tolist())
    assert labels == {H, A, D, X}
    # the untracked frame (6) becomes X, the frames without a group D
    t = np.arange(48) / 25 + 0.1  # source time of the first half's 25 Hz frames
    assert (m["label"][first][np.isclose(t, 0.6)] == X).all()
    assert (m["label"][first][(t >= 1.0) & (t < 1.4)] == D).all()
    # the ball is kept only where it was detected
    near_3 = (t > 0.2) & (t < 0.4)
    assert np.isnan(m["bx"][first][near_3]).all()
    assert m["px"].dtype == np.float32 and m["label"].dtype == np.int8


def test_every_file_is_pinned():
    hashes = sc.expected_hashes()
    ids = sc.match_ids()
    assert len(ids) == 20 and len(hashes) == 40
    for mid in ids:
        assert {f"{mid}_tracking_extrapolated.jsonl", f"{mid}_match.json"} <= set(hashes)
    assert all(len(h) == 64 for h in hashes.values())
    assert sc.COMMIT in sc.RAW_URL and sc.COMMIT in sc.LFS_URL
