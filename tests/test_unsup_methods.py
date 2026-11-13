"""Focused tests of the label-free methods, on small synthetic matches."""

import numpy as np

from possession import features
from possession.match import AWAY, HOME, Match
from possession.stages.ball_state import stoppage_evidence
from possession.stages.decode import switch_cost
from possession.stages.team import control_evidence

FPS = 25


def _match(ball_xy, players_xy, speeds, teams=(HOME, AWAY)):
    """ball_xy (n, 2), players_xy (n, S, 2), speeds (n, S); one graded half."""
    n = len(ball_xy)
    ball = np.concatenate([ball_xy, np.full((n, 1), np.nan)], axis=1)
    players = np.concatenate([players_xy, speeds[..., None]], axis=2)
    return Match(
        game="SYNTH", fps=FPS, pitch=(105.0, 68.0), ball=ball, players=players,
        team_of_track=np.array(teams, dtype=np.int8), half=np.ones(n, dtype=np.int8),
    )  # fmt: skip


def _still_players(n, speed):
    xy = np.tile(np.array([[30.0, 30.0], [70.0, 30.0]]), (n, 1, 1))
    return xy, np.full((n, 2), speed)


def test_stoppage_evidence_reads_visibility_speed_and_place():
    n = 200
    line = np.stack([np.linspace(20, 80, n), np.full(n, 34.0)], axis=1)  # a ball in open play
    fast_xy, fast = _still_players(n, 4.0)
    slow_xy, slow = _still_players(n, 1.0)

    seen_fast = stoppage_evidence(_match(line, fast_xy, fast))
    assert seen_fast[50:150].min() > 0.9  # tracked ball, players running: live

    gone = line.copy()
    gone[:] = np.nan
    gone_slow = stoppage_evidence(_match(gone, slow_xy, slow))
    gone_fast = stoppage_evidence(_match(gone, fast_xy, fast))
    assert gone_slow.max() < 0.1  # no ball, nobody running: a stoppage
    assert gone_fast.min() > gone_slow.max()  # no ball but players sprinting: far less sure

    outside = np.stack([np.linspace(20, 80, n), np.full(n, -3.0)], axis=1)  # beyond the touchline
    assert stoppage_evidence(_match(outside, fast_xy, fast))[50:150].max() < seen_fast[50:150].min()

    placed = np.tile(np.array([[50.0, 34.0]]), (n, 1))  # a ball that does not move
    assert stoppage_evidence(_match(placed, fast_xy, fast))[50:].max() < seen_fast[50:150].min()
    assert not np.isnan(seen_fast).any() and not np.isnan(gone_slow).any()


def test_control_evidence_carrier_duel_and_passing_ball():
    n = 100
    t = np.arange(n) / FPS
    # a home player dribbles: the ball stays 0.5 m ahead of him at his speed
    home = np.stack([20 + 3.0 * t, np.full(n, 30.0)], axis=1)
    away = np.tile(np.array([[80.0, 30.0]]), (n, 1))
    players = np.stack([home, away], axis=1)
    speeds = np.stack([np.full(n, 3.0), np.zeros(n)], axis=1)
    carried = control_evidence(_match(home + [0.5, 0.0], players, speeds))
    assert np.nanmin(carried[10:90]) > 0.9

    mirrored = control_evidence(_match(home + [0.5, 0.0], players, speeds, teams=(AWAY, HOME)))
    np.testing.assert_allclose(mirrored[10:90], 1 - carried[10:90], atol=1e-6)  # symmetric in team

    nobody = control_evidence(_match(home + [10.0, 10.0], players, speeds))
    assert np.isnan(nobody).all()  # no player near the ball: no evidence

    both = np.stack([home, home + [1.0, 0.0]], axis=1)
    duel = control_evidence(_match(home + [0.5, 0.0], both, speeds))
    assert np.isnan(duel[10:90]).all()  # both teams on the ball: a duel says nothing

    # the same geometry, but the ball flies past a standing home player at 20 m/s
    standing = np.tile(np.array([[50.0, 30.0]]), (n, 1))
    flying = np.stack([30 + 20.0 * t, np.full(n, 30.5)], axis=1)
    passing = control_evidence(_match(flying, np.stack([standing, away], axis=1), np.zeros((n, 2))))
    near = ~np.isnan(passing)
    assert near.any() and passing[near].max() < 0.7  # weak evidence, far from the carrier's 0.9+


def test_nearest_by_team_breaks_ties_low_and_marks_a_missing_team():
    n = 3
    ball = np.tile(np.array([[50.0, 34.0]]), (n, 1))
    # slots: home, home, away, away; both home players 2 m from the ball (a tie)
    xy = np.tile(np.array([[52.0, 34.0], [48.0, 34.0], [50.0, 37.0], [60.0, 34.0]]), (n, 1, 1))
    xy[1, 2:] = np.nan  # frame 1: the whole away team is missing
    xy[2] = np.nan  # frame 2: nobody is tracked
    match = _match(ball, xy, np.zeros((n, 4)), teams=(HOME, HOME, AWAY, AWAY))
    near = features.nearest_by_team(match)
    np.testing.assert_allclose(near["distance"][0], [2.0, 3.0])
    np.testing.assert_array_equal(near["slot"][0], [0, 2])  # the tie goes to the lowest slot
    assert near["slot"][1, 1] == -1 and np.isnan(near["distance"][1, 1])
    assert near["slot"][1, 0] == 0
    assert (near["slot"][2] == -1).all() and np.isnan(near["distance"][2]).all()


def _decode(alive, home, **params):
    n = len(alive)
    xy, speeds = _still_players(n, 3.0)
    match = _match(np.zeros((n, 2)), xy, speeds)
    return switch_cost(match, np.asarray(alive, dtype=np.float32), np.asarray(home), **params)


def test_switch_cost_ignores_a_blip_and_places_a_real_change_on_time():
    n = 500
    alive = np.full(n, 0.95)
    home = np.full(n, 0.9)
    home[200] = 0.02  # one frame of the other team
    labels = _decode(alive, home)
    assert set(labels) == {"H"}  # a single-frame flip does not create a possession

    home[300:] = 0.1  # a sustained change of control
    labels = _decode(alive, home)
    change = int(np.flatnonzero(labels == "A")[0])
    assert (
        abs(change - 300) <= 3 and set(labels[change:]) == {"A"} and set(labels[:change]) == {"H"}
    )


def test_switch_cost_keeps_the_owner_without_evidence_and_never_abstains():
    n = 600
    alive = np.full(n, 0.95)
    home = np.full(n, np.nan)
    home[:100] = 0.9  # home in control, then 12 s with no team evidence at all, then home again
    home[400:] = 0.9
    alive[150:300] = 0.02  # a stoppage in the middle
    labels = _decode(alive, home)
    assert "X" not in labels
    assert (
        set(labels[:150]) == {"H"} and set(labels[150:300]) == {"D"} and set(labels[300:]) == {"H"}
    )

    alive[150:153] = 0.95
    alive[160] = 0.95  # flicker at the edge of the stoppage does not split it
    assert (np.diff((_decode(alive, home) == "D").astype(int)) != 0).sum() == 2


# -- second version: kick-off set-ups, tie rule, edge snapping, touch episodes -----------------


def _two_teams(n, home_x, away_x):
    """8 players a side standing on given x positions (each (n,) or scalar), spread over y."""
    xs = [np.broadcast_to(np.asarray(home_x, dtype=float), (n,))] * 8
    xs += [np.broadcast_to(np.asarray(away_x, dtype=float), (n,))] * 8
    ys = [5.0 + 8.0 * i for i in range(8)] * 2
    xy = np.stack(
        [np.stack([x, np.full(n, y)], axis=1) for x, y in zip(xs, ys, strict=True)], axis=1
    )
    return xy, tuple([HOME] * 8 + [AWAY] * 8)


def _kickoff_match(n=600, kick=200, cross=260, ball_seen=True):
    """Both teams in their own halves until `cross`; the ball leaves the centre mark at `kick`."""
    t = np.arange(n)
    home_x = np.where(t < cross, 40.0, 60.0)  # home defends the low-x side, then pushes over
    away_x = np.full(n, 70.0)
    xy, teams = _two_teams(n, home_x, away_x)
    ball = np.tile(np.array([52.5, 34.0]), (n, 1))
    ball[kick:, 0] = 52.5 - 0.4 * (t[kick:] - kick + 1)  # 10 m/s towards the home half
    ball[:, 0] = np.clip(ball[:, 0], 5.0, None)
    if not ball_seen:
        ball[:] = np.nan
    return _match(ball, xy, np.full((n, 16), 1.0), teams=teams)


def test_kickoff_setup_and_kick_time():
    from possession import features

    setups = features.kickoffs(_kickoff_match())
    assert setups["start"].tolist() == [0] and setups["stop"].tolist() == [260]
    assert 200 <= setups["kick"][0] <= 210  # the ball left the centre mark at frame 200

    unseen = features.kickoffs(_kickoff_match(ball_seen=False))
    assert unseen["kick"].tolist() == [260 - 45]  # no ball: 1.8 s before the set-up ends

    xy, teams = _two_teams(300, 60.0, 70.0)  # home already across the line: open play
    open_play = _match(np.full((300, 2), 50.0), xy, np.full((300, 16), 3.0), teams=teams)
    assert len(features.kickoffs(open_play)["start"]) == 0


def test_stoppage_evidence_is_dead_until_the_kick():
    with_rule = stoppage_evidence(_kickoff_match())
    without = stoppage_evidence(_kickoff_match(), kickoff_llr=0.0)
    assert with_rule[:195].max() < 0.01  # waiting for the kick-off, however visible the ball
    assert with_rule[:195].max() < without[:195].min()
    np.testing.assert_allclose(with_rule[300:], without[300:])  # nothing changes after it


def test_wait_before_the_first_setup_of_a_half_is_dead():
    n, lead = 900, 300
    late = _kickoff_match(n=n - lead, kick=200, cross=260)
    xy = np.concatenate([np.tile(late.players[:1, :, :2], (lead, 1, 1)), late.players[:, :, :2]])
    xy[:lead, :8, 0] = 60.0  # before the set-up the home players stand in the other half
    ball = np.concatenate([np.tile(late.ball[:1, :2], (lead, 1)), late.ball[:, :2]])
    walking = _match(ball, xy, np.full((n, 16), 1.0), teams=tuple(late.team_of_track))
    evidence = stoppage_evidence(walking)
    assert evidence[:lead].max() < 0.01  # slow players before the first set-up: not yet started
    running = _match(ball, xy, np.full((n, 16), 4.0), teams=tuple(late.team_of_track))
    assert stoppage_evidence(running)[:lead].min() > 0.5  # players running: the half is under way


def test_decoder_switches_at_the_first_evidence_for_either_team():
    n = 400
    alive = np.full(n, 0.95)
    for first, second in ((0.9, 0.1), (0.1, 0.9)):  # home then away, and the mirror image
        home = np.full(n, np.nan)
        home[:100] = first
        home[250:] = second  # six seconds without any team evidence in between
        labels = _decode(alive, home)
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        assert changes.tolist() == [250], (first, second)


def test_stoppage_edges_snap_to_the_ball_gap():
    from possession.stages.decode import snap_stoppage_edges

    n = 1000
    t = np.arange(n)
    ball = np.stack([20.0 + 0.4 * t, np.full(n, 30.0)], axis=1)  # a ball moving at 10 m/s
    ball[300:500] = np.nan  # the gap
    xy, speeds = _still_players(n, 2.0)
    match = _match(ball, xy, speeds)
    labels = np.full(n, "H", dtype="<U1")
    labels[330:515] = "D"  # decoded 30 frames late at the start, 15 late at the end
    snapped = snap_stoppage_edges(match, labels, start_s=3.0, end_s=3.0, restart_speed_m_s=3.0)
    dead = np.flatnonzero(snapped == "D")
    assert dead[0] == 300 and dead[-1] == 499  # both edges on the gap's edges

    early = np.full(n, "H", dtype="<U1")
    early[330:480] = "D"  # restart decoded while the ball is still missing
    dead = np.flatnonzero(snap_stoppage_edges(match, early, 3.0, 3.0, 3.0) == "D")
    assert dead[-1] == 499

    far = np.full(n, "H", dtype="<U1")
    far[420:600] = "D"  # begins 120 frames after the ball was lost: left alone
    kept = snap_stoppage_edges(match, far, start_s=3.0, end_s=3.0, restart_speed_m_s=3.0)
    assert np.flatnonzero(kept == "D")[0] == 420 and np.flatnonzero(kept == "D")[-1] == 599

    placed = ball.copy()
    placed[500:, 0] = placed[500, 0]  # the ball reappears and does not move: no restart yet
    still = snap_stoppage_edges(_match(placed, xy, speeds), labels, 3.0, 3.0, 3.0)
    assert np.flatnonzero(still == "D")[-1] == 514
    np.testing.assert_array_equal(
        snap_stoppage_edges(match, labels, 0.0, 0.0, 3.0), labels
    )  # zero tolerances switch it off


def _pass_match(receiver_team, gap=None):
    """A home player holds the ball, passes it 20 m in one second, and a player receives it."""
    from possession.stages.team import touch_episodes

    n = 200
    passer = np.tile(np.array([20.0, 30.0]), (n, 1))
    receiver = np.tile(np.array([40.5, 30.0]), (n, 1))
    far = np.tile(np.array([90.0, 60.0]), (n, 1))
    ball = np.tile(np.array([20.5, 30.0]), (n, 1))  # at the passer's feet
    flight = np.arange(75, 100)
    ball[flight, 0] = 20.5 + 0.8 * (flight - 74)  # 20 m/s
    ball[100:, 0] = 40.5  # at the receiver's feet
    if gap:
        ball[gap[0] : gap[1]] = np.nan
    players = np.stack([passer, receiver, far], axis=1)
    teams = (HOME, receiver_team, AWAY if receiver_team == HOME else HOME)
    match = _match(ball, players, np.zeros((n, 3)), teams=teams)
    return touch_episodes(match)


def test_touch_episodes_bridge_a_completed_pass_only():
    completed = _pass_match(HOME)
    assert np.nanmin(completed[10:70]) > 0.9 and np.nanmin(completed[110:190]) > 0.9
    assert np.nanmin(completed[80:97]) > 0.9  # in flight between two home touches: still home

    intercepted = _pass_match(AWAY)
    assert np.nanmin(intercepted[10:70]) > 0.9 and np.nanmax(intercepted[110:190]) < 0.1
    assert np.isnan(intercepted[80:97]).all()  # in flight towards the other team: no claim

    lost = _pass_match(HOME, gap=(85, 90))
    assert np.isnan(lost[80:97]).all()  # the ball was not tracked all the way: no claim


def test_touch_episodes_ignore_a_ball_flying_past():
    from possession.stages.team import touch_episodes

    n = 100
    t = np.arange(n) / FPS
    flying = np.stack([10 + 20.0 * t, np.full(n, 30.5)], axis=1)  # straight line, 20 m/s
    standing = np.tile(np.array([50.0, 30.0]), (n, 1))
    away = np.tile(np.array([90.0, 60.0]), (n, 1))
    match = _match(flying, np.stack([standing, away], axis=1), np.zeros((n, 2)))
    assert np.isnan(touch_episodes(match)).all()  # near a player, but neither carried nor played

    # the same ball deflected by that player (it comes back) is a touch
    deflected = flying.copy()
    turn = int(np.argmin(np.abs(flying[:, 0] - 50.0)))
    deflected[turn:, 0] = flying[turn, 0] - 20.0 * (t[turn:] - t[turn])
    touched = touch_episodes(
        _match(deflected, np.stack([standing, away], axis=1), np.zeros((n, 2)))
    )
    assert np.nanmax(touched) > 0.9


def test_a_moving_ball_inside_the_pitch_is_evidence_of_play():
    n = 200
    xy, slow = _still_players(n, 1.0)  # nobody running, as just after a restart
    t = np.arange(n) / FPS
    rolling = np.stack([20.0 + 1.0 * t, np.full(n, 34.0)], axis=1)  # 1 m/s
    played = np.stack([20.0 + 8.0 * t, np.full(n, 34.0)], axis=1)  # 8 m/s
    assert (
        stoppage_evidence(_match(played, xy, slow))[50:150].min()
        > stoppage_evidence(_match(rolling, xy, slow))[50:150].max()
    )
    outside = np.stack([20.0 + 8.0 * t, np.full(n, -2.0)], axis=1)  # fast, but off the pitch
    assert stoppage_evidence(_match(outside, xy, slow))[50:150].max() < 0.5


def test_restart_goes_to_the_team_standing_at_the_ball():
    from possession.stages.team import touch_episodes

    n = 300
    t = np.arange(n)
    taker = np.tile(np.array([0.5, 0.5]), (n, 1))  # at the corner flag
    defender = np.tile(np.array([12.0, 30.0]), (n, 1))
    ball = np.full((n, 2), np.nan)  # no ball for the first 100 frames: the stoppage
    flight = t >= 100
    ball[flight, 0] = 1.0 + 0.5 * (t[flight] - 100)  # the corner is in the air at 12.5 m/s
    ball[flight, 1] = 1.0 + 0.5 * (t[flight] - 100)
    players = np.stack([taker, defender], axis=1)
    for taker_team in (HOME, AWAY):
        teams = (taker_team, AWAY if taker_team == HOME else HOME)
        evidence = touch_episodes(_match(ball, players, np.zeros((n, 2)), teams=teams))
        claim = evidence[100:140] if taker_team == HOME else 1.0 - evidence[100:140]
        assert not np.isnan(claim).any() and claim.min() >= 0.9 - 1e-6  # the taker's team
        assert np.isnan(evidence[:100]).all()
        off = touch_episodes(_match(ball, players, np.zeros((n, 2)), teams=teams), restart_p=0.5)
        assert np.isnan(off[105:140]).all()  # without the rule a ball in flight says nothing

    close = np.stack([taker, np.tile(np.array([2.0, 2.0]), (n, 1))], axis=1)  # opponent right there
    contested = touch_episodes(_match(ball, close, np.zeros((n, 2))))
    assert np.isnan(contested[100:103]).all()  # no clear taker: no claim
