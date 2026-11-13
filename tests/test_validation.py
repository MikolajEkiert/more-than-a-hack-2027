"""The external-match harness, on synthetic inputs: no download, no real data."""

import xml.etree.ElementTree as ET

import numpy as np

from possession import validation
from possession.match import AWAY, HOME, Match

META = """<root>
  <PlayerChannels>
    <PlayerChannel channelId="x" id="p1_x" playerId="A1"/><PlayerChannel channelId="y" id="p1_y" playerId="A1"/>
    <PlayerChannel channelId="x" id="p2_x" playerId="B1"/><PlayerChannel channelId="y" id="p2_y" playerId="B1"/>
    <PlayerChannel channelId="x" id="p3_x" playerId="A2"/><PlayerChannel channelId="y" id="p3_y" playerId="A2"/>
  </PlayerChannels>
  <DataFormatSpecification startFrame="1" endFrame="2">
    <PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
    <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
  </DataFormatSpecification>
  <DataFormatSpecification startFrame="3" endFrame="3">
    <PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/>
    <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
  </DataFormatSpecification>
</root>"""  # noqa: E501


def _event(index, kind, team, start, end, period=1):
    return {
        "index": index,
        "type": {"name": kind},
        "team": {"id": team},
        "start": {"frame": start},
        "end": {"frame": end},
        "period": period,
    }


def test_tracking_follows_the_column_layout_across_a_substitution(tmp_path):
    layout = validation.column_layout(ET.fromstring(META), ["A1", "A2", "B1"])
    assert layout == [(1, 2, [0, 2]), (3, 3, [1, 2])]
    raw = tmp_path / "tracking.txt"
    raw.write_text(
        "1:0.1,0.2;0.5,0.6:0.3,0.3\n2:0.1,0.2;0.5,0.6:NaN,NaN\n3:0.9,0.9;0.5,0.6:0.4,0.4\n"
    )
    ball, players = validation.parse_tracking(raw, layout, n_slots=3)
    np.testing.assert_allclose(players[0, 0], [0.1, 0.2])  # A1 on the pitch
    assert np.isnan(players[0, 1]).all()  # A2 not yet
    np.testing.assert_allclose(players[2, 1], [0.9, 0.9])  # A2 came on in the first column
    assert np.isnan(players[2, 0]).all()  # A1 went off
    np.testing.assert_allclose(players[:, 2, 0], 0.5)  # B1 throughout
    assert np.isnan(ball[1]).all() and ball[2].tolist() == [0.4, 0.4]


def test_labels_follow_the_provider_definition():
    events = [
        _event(1, "SET PIECE", "HOME", 11, 11),
        _event(2, "PASS", "HOME", 20, 30),
        _event(3, "BALL LOST", "HOME", 40, 40),  # stays with the home team
        _event(4, "CHALLENGE", "AWAY", 42, 42),  # a challenge is not possession
        _event(5, "RECOVERY", "AWAY", 51, 51),
        _event(6, "PASS", "AWAY", 90, 100),
    ]
    owner, half = validation.event_labels(events, n_frames=120, home_id="HOME")
    assert (owner[:10] == -1).all() and (owner[10:50] == HOME).all() and (owner[50:] == AWAY).all()
    assert (half[:10] == 0).all() and (half[10:100] == 1).all() and (half[100:] == 0).all()

    seen = np.ones(120, dtype=bool)
    seen[60:80] = False  # a stoppage
    seen[30:32] = False  # a two-frame dropout: not a stoppage
    labels = validation.labels_from(seen, owner, half)
    assert set(labels[:10]) == {"X"} and set(labels[10:50]) == {"H"}
    assert (
        set(labels[50:60]) == {"A"} and set(labels[60:80]) == {"D"} and set(labels[80:100]) == {"A"}
    )


def _clean_match(n=6000, slots=22, fps=25):
    rng = np.random.default_rng(1)
    xy = np.cumsum(rng.normal(0, 0.1, (n, slots, 2)), axis=0) + rng.uniform(20, 50, (1, slots, 2))
    players = np.concatenate([xy, validation.speeds(xy, fps)[..., None]], axis=2)
    ball = np.concatenate([xy[:, 0], np.full((n, 1), np.nan)], axis=1)
    labels = np.full(n, "H", dtype="<U1")
    labels[3000:] = "A"
    for start in (1000, 2500, 4500):
        labels[start : start + 300] = "D"
        ball[start : start + 300] = np.nan
    match = Match(
        game="SYNTH", fps=fps, pitch=(105.0, 68.0), ball=ball, players=players,
        team_of_track=np.array([HOME, AWAY] * (slots // 2), dtype=np.int8),
        half=np.ones(n, dtype=np.int8),
    )  # fmt: skip
    return match, labels


def test_degrade_is_seeded_pure_and_does_what_it_says():
    match, labels = _clean_match()
    before = match.ball.copy()
    a = validation.degrade(match, labels, seed=3, lost_ball_share=0.2, players_visible=20.0)
    b = validation.degrade(match, labels, seed=3, lost_ball_share=0.2, players_visible=20.0)
    c = validation.degrade(match, labels, seed=4, lost_ball_share=0.2, players_visible=20.0)
    np.testing.assert_array_equal(match.ball, before)  # the input is not mutated
    np.testing.assert_array_equal(a.ball, b.ball)
    np.testing.assert_array_equal(a.players, b.players)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)

    alive = labels != "D"
    lost = (~a.ball_seen[alive]).mean()
    assert 0.15 <= lost <= 0.35  # asked for 0.2; whole stretches overshoot a little
    visible = (~np.isnan(a.players[:, :, 0])).sum(axis=1).mean()
    assert 18.5 <= visible <= 21.5
    assert a._cache == {} and a.n_frames == match.n_frames


def test_degrade_smooths_the_ball_only_within_tracked_stretches():
    n, fps = 400, 25
    x = np.where(np.arange(n) < 200, 0.0, 10.0)  # a kick: the ball jumps 10 m in one frame
    x[300:350] = np.nan  # a gap in the tracking
    ball = np.stack([x, np.full(n, 34.0), np.full(n, np.nan)], axis=1)
    players = np.full((n, 2, 3), 1.0)
    match = Match(
        game="SYNTH", fps=fps, pitch=(105.0, 68.0), ball=ball, players=players,
        team_of_track=np.array([HOME, AWAY], dtype=np.int8), half=np.ones(n, dtype=np.int8),
    )  # fmt: skip
    labels = np.full(n, "H", dtype="<U1")
    only_smoothing = dict(
        lost_ball_share=0.0, visible_dead_prob=0.0, ball_offset_m=0.0, players_visible=99.0
    )
    out = validation.degrade(match, labels, ball_smooth_s=0.6, **only_smoothing)
    bx = out.ball[:, 0]
    assert np.isnan(bx[300:350]).all() and not np.isnan(np.delete(bx, range(300, 350))).any()
    np.testing.assert_allclose(bx[:180], 0.0)  # far from the kick: unchanged
    np.testing.assert_allclose(bx[220:300], 10.0)  # next to the gap: not pulled towards 0
    np.testing.assert_allclose(bx[350:], 10.0)
    blurred = bx[190:210]
    assert ((blurred > 0) & (blurred < 10)).sum() >= 10 and np.all(np.diff(blurred) >= 0)

    raw = validation.degrade(match, labels, ball_smooth_s=0.0, **only_smoothing)
    np.testing.assert_array_equal(raw.ball, match.ball)


def test_score_uses_the_official_metric():
    _, labels = _clean_match()
    assert validation.score(labels, labels)["spellF1"] == 1.0
    late = np.roll(labels, 25)
    assert 0 < validation.score(late, labels)["spellF1"] < 1.0
    constant = np.full(len(labels), "H", dtype="<U1")
    assert validation.score(constant, labels)["n_pred_possessions"] == 1


def test_record_keeps_the_external_score(tmp_path, monkeypatch):
    from possession import config as config_mod
    from possession import experiments

    monkeypatch.setattr(experiments, "RUNS_DIR", tmp_path)
    cfg = config_mod.load("configs/vote_smooth25.toml")
    external = {"game": "metrica-sg3", "condition": "degraded(seed=0)", "spellF1": 0.1234}
    experiments.record(cfg, "", {"n_possessions": 7}, None, 0.0, external=external)
    (entry,) = experiments.load_all()
    assert entry["external"] == external


def test_table_shows_the_external_score_only_where_there_is_one():
    from possession import experiments

    external = {"spellF1": 0.1234}
    lines = experiments.table([{"name": "a", "external": external}, {"name": "b"}]).splitlines()
    assert "ext spF1" in lines[0]
    assert "0.1234" in lines[2]
    assert "0.1234" not in lines[3]


def test_table_marks_clean_and_degraded_external_scores():
    from possession import experiments

    entries = [
        {"name": "a", "external": {"spellF1": 0.1234, "condition": "clean"}},
        {"name": "b", "external": {"spellF1": 0.4321, "condition": "degraded(seed=0)"}},
    ]
    lines = experiments.table(entries).splitlines()
    assert "0.1234c" in lines[2]
    assert "0.4321d" in lines[3]
