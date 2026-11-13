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
