"""The external-match harness, on synthetic inputs: no download, no real data."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

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


def _event(order, kind, team, start, end=None, tokens=(), period=1):
    """A `validation.Event` with 0-based frames."""
    end = start if end is None else end
    return validation.Event(start, end, team, kind, tuple(tokens), period, order)


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


def test_events_read_the_same_from_both_file_formats(tmp_path):
    as_json = tmp_path / "events.json"
    as_json.write_text(
        '{"data": ['
        '{"index": 2, "team": {"id": "TB"}, "type": {"name": "RECOVERY"},'
        ' "subtypes": {"name": "INTERCEPTION"}, "start": {"frame": 51}, "end": {"frame": 51},'
        ' "period": 1},'
        '{"index": 1, "team": {"id": "TA"}, "type": {"name": "Ball Lost"},'
        ' "subtypes": [{"name": "HEAD"}, {"name": "INTERCEPTION"}],'
        ' "start": {"frame": 11}, "end": {"frame": 50}, "period": 1}]}'
    )
    as_csv = tmp_path / "events.csv"
    as_csv.write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To\n"
        "Home,BALL LOST,HEAD-INTERCEPTION,1,11,0.4,50,2.0,Player1,\n"
        "Away,RECOVERY,INTERCEPTION,1,51,2.04,0,0,Player15,\n"
    )
    from_json = validation.events_from_json(as_json, home_id="TA")
    from_csv = validation.events_from_csv(as_csv)
    assert [e.kind for e in from_json] == ["BALL LOST", "RECOVERY"]  # sorted by frame
    for a, b in zip(from_json, from_csv, strict=True):
        assert (a.start, a.end, a.team, a.kind, a.tokens) == (
            b.start,
            b.end,
            b.team,
            b.kind,
            b.tokens,
        )
    assert from_json[0].start == 10 and from_json[0].tokens == ("HEAD", "INTERCEPTION")
    assert from_csv[1].end == from_csv[1].start == 50  # an end before the start is clamped


HEAD_TENNIS = [
    _event(1, "SET PIECE", HOME, 10, tokens=("KICK OFF",)),
    _event(2, "PASS", HOME, 20, 30),
    _event(3, "BALL LOST", HOME, 40, 49, tokens=("INTERCEPTION",)),
    _event(4, "CHALLENGE", AWAY, 45),  # a challenge is never possession
    _event(5, "RECOVERY", AWAY, 50, tokens=("INTERCEPTION",)),  # a header ...
    _event(6, "BALL LOST", AWAY, 50, 59, tokens=("HEAD", "INTERCEPTION")),  # ... straight back
    _event(7, "RECOVERY", HOME, 60),
    _event(8, "BALL LOST", HOME, 60, 69, tokens=("HEAD",)),
    _event(9, "RECOVERY", AWAY, 70),  # this one is followed by a pass
    _event(10, "PASS", AWAY, 90, 100),
    _event(11, "RECOVERY", HOME, 200),  # held for more than MIN_HELD_FRAMES, then lost
    _event(12, "BALL LOST", HOME, 230, 240),
    _event(13, "RECOVERY", AWAY, 241),  # a save that goes straight out
    _event(14, "BALL OUT", AWAY, 242, 250),
    _event(15, "BALL LOST", HOME, 300, 320, tokens=("GOAL KICK",)),  # a goal kick is a restart
]


def _owner_changes(owner):
    frames = np.flatnonzero(np.diff(owner.astype(int)) != 0) + 1
    return [(int(f), int(owner[f])) for f in frames]


def test_touch_convention_flips_on_every_on_ball_event():
    owner = validation.event_owner(HEAD_TENNIS, 400, "touch")
    assert _owner_changes(owner) == [
        (10, HOME), (50, AWAY), (60, HOME), (70, AWAY), (200, HOME), (241, AWAY), (300, HOME),
    ]  # fmt: skip


def test_controlled_convention_needs_a_hold_or_an_action():
    owner = validation.event_owner(HEAD_TENNIS, 400, "controlled")
    # the two headers and the save never become possessions; the recovery followed by a pass
    # and the one held for 30 frames do, each dated from its first touch
    assert _owner_changes(owner) == [(10, HOME), (70, AWAY), (200, HOME)]
    assert (owner[:10] == -1).all()
    with pytest.raises(ValueError):
        validation.event_owner(HEAD_TENNIS, 400, "sticky")


def test_halves_come_from_the_events():
    events = [
        _event(1, "SET PIECE", HOME, 10),
        _event(2, "PASS", HOME, 90, 99),
        _event(3, "SET PIECE", AWAY, 150, period=2),
        _event(4, "PASS", AWAY, 180, 189, period=2),
    ]
    half = validation.event_halves(events, 200)
    assert (half[:10] == 0).all() and (half[10:100] == 1).all()
    assert (half[100:190] == 2).all() and (half[190:] == 0).all()  # the wait for kick-off is graded


def test_a_stretch_without_a_ball_is_a_stoppage_only_if_confirmed():
    seen = np.ones(700, dtype=bool)
    stretches = {
        "dropout": (30, 32),
        "restart": (60, 80),
        "unexplained": (100, 140),
        "stopper": (150, 165),
        "long": (200, 300),
        "restart_inside": (400, 430),
    }
    for start, stop in stretches.values():
        seen[start:stop] = False
    events = [
        _event(1, "SET PIECE", HOME, 80, tokens=("THROW IN",)),  # at the end of "restart"
        _event(2, "BALL OUT", AWAY, 145, 149),  # ends where "stopper" begins
        _event(3, "PASS", HOME, 425, 440, tokens=("GOAL KICK",)),  # inside "restart_inside"
        _event(4, "PASS", HOME, 141, 143),  # an ordinary pass next to "unexplained"
    ]
    dead = validation.stoppage_mask(seen, events, fps=25)
    for name, (start, stop) in stretches.items():
        expected = name not in ("dropout", "unexplained")
        assert dead[start:stop].all() == expected and dead[start:stop].any() == expected, name
    assert not dead[seen].any()

    owner = np.full(700, HOME, dtype=np.int8)
    owner[:5] = -1
    half = np.ones(700, dtype=np.int8)
    half[650:] = 0
    labels = validation.labels_from(owner, half, dead)
    assert set(labels[:5]) == {"D"} and set(labels[100:140]) == {"H"}  # a dropout stays live
    assert set(labels[200:300]) == {"D"} and set(labels[650:]) == {"X"}


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


def test_degradation_profiles_are_files_and_can_be_overridden():
    assert "label_free" in validation.degradation_profiles()
    params = validation.degradation_profile("label_free")
    assert set(params) == set(validation.DEGRADATION_KEYS)
    match, labels = _clean_match()
    same = validation.degrade(match, labels, seed=1, profile="label_free")
    default = validation.degrade(match, labels, seed=1)
    np.testing.assert_array_equal(same.ball, default.ball)
    untouched = validation.degrade(
        match, labels, seed=1, lost_ball_share=0.0, visible_dead_prob=0.0, ball_offset_m=0.0,
        ball_smooth_s=0.0, players_visible=99.0,
    )  # fmt: skip
    np.testing.assert_array_equal(untouched.ball, match.ball)  # every amount set to nothing
    with pytest.raises(TypeError):
        validation.degrade(match, labels, ball_ofset_m=1.0)  # a typo is an error, not a no-op
    with pytest.raises(KeyError):
        validation.degrade(match, labels, profile="no_such_profile")


def test_baseline_family_covers_the_published_windows():
    match, _ = _clean_match()
    family = validation.baseline_family(match)
    assert set(family) == set(validation.PUBLISHED_FULL_MATCH) == {0, 25, 50, 125}
    for labels in family.values():
        assert labels.shape == (match.n_frames,) and set(labels) <= {"H", "A", "D", "X"}
    # the raw baseline flickers more than the smoothed ones
    changes = {w: int((labels[1:] != labels[:-1]).sum()) for w, labels in family.items()}
    assert changes[0] >= changes[25] >= changes[125]


def _cells(scores):
    """{(game, convention): spellF1} -> evaluate()-shaped cells."""
    return [
        {"game": game, "seed": 0, "convention": convention, "spellF1": value}
        for (game, convention), value in scores.items()
    ]


def test_acceptance_rule():
    games, conventions = ("g1", "g2", "g3"), validation.CONVENTIONS
    reference = _cells({(g, c): 0.50 for g in games for c in conventions})
    better = _cells({(g, c): 0.55 for g in games for c in conventions})
    verdict = validation.compare(better, reference)
    assert verdict["accepted"] and verdict["gain"] == 0.05 and verdict["games_won"] == 3

    small = _cells({(g, c): 0.51 for g in games for c in conventions})
    assert not validation.compare(small, reference)["accepted"]  # below MIN_GAIN

    lopsided = _cells(
        {(g, "touch"): 0.40 for g in games} | {(g, "controlled"): 0.70 for g in games}
    )
    verdict = validation.compare(lopsided, reference)
    assert verdict["gain"] == 0.05 and not verdict["accepted"]  # loses under one convention

    one_game = _cells(
        {("g1", c): 0.80 for c in conventions}
        | {(g, c): 0.49 for g in games[1:] for c in conventions}
    )
    verdict = validation.compare(one_game, reference)
    assert verdict["gain"] > validation.MIN_GAIN and verdict["games_won"] == 1
    assert not verdict["accepted"]  # one match carries it

    summary = validation.summarise(better)
    assert summary["mean"] == 0.55 and summary["by_game"]["g2"] == 0.55
    assert summary["by_game_convention"]["g3"]["controlled"] == 0.55


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
