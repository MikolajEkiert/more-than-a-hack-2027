"""The decoder options of task 14: each is off by default and leaves the output byte-identical."""

import numpy as np
import pytest

from possession import features
from possession.match import AWAY, HOME, Match
from possession.stages.decode import (
    _viterbi3,
    shift_changes,
    short_spells,
    short_stoppages,
    snap_stoppage_edges,
    snap_to_line_exit,
    snap_to_reentry,
    switch_cost,
)
from tests.conftest import make_match

FPS = 25
V1 = {"turnover_cost": 5.0, "stop_cost": 12.0, "restart_cost": 12.0}
V1_SNAP = {**V1, "snap_start_s": 3.0, "snap_end_s": 3.0}


def _previous_switch_cost(
    match, p_alive, p_home, turnover_cost=3.0, stop_cost=8.0, restart_cost=8.0,
    frames_per_observation=5.0, team_weight=1.0, alive_weight=1.0, snap_start_s=0.0,
    snap_end_s=0.0, restart_speed_m_s=3.0,
):  # fmt: skip
    """`switch_cost` as it was before task 14, copied verbatim (the reference of the default)."""
    eps = 1e-6
    alive = np.clip(np.asarray(p_alive, dtype=np.float64), eps, 1 - eps)
    home = np.asarray(p_home, dtype=np.float64)
    known = ~np.isnan(home)
    home = np.clip(np.where(known, home, 0.5), eps, 1 - eps)
    scale = 1.0 / frames_per_observation
    live = alive_weight * scale * np.log(alive)
    emission = np.stack(
        [
            live + team_weight * scale * np.where(known, np.log(home), 0.0),
            live + team_weight * scale * np.where(known, np.log(1 - home), 0.0),
            alive_weight * scale * np.log(1 - alive),
        ],
        axis=1,
    )
    cost = [
        [0.0, turnover_cost, stop_cost],
        [turnover_cost, 0.0, stop_cost],
        [restart_cost, restart_cost, 0.0],
    ]
    names = np.array(["H", "A", "D"])
    labels = np.full(match.n_frames, "X", dtype="<U1")
    for number in (1, 2):
        for start, stop in zip(*features.runs(match.half == number), strict=True):
            labels[start:stop] = names[_viterbi3(emission[start:stop], cost)]
    if snap_start_s > 0 or snap_end_s > 0:
        labels = snap_stoppage_edges(match, labels, snap_start_s, snap_end_s, restart_speed_m_s)
    return labels


def _evidence(match, seed=0):
    """Slowly varying p_alive and p_home: a stoppage every 40 s, flips of the team evidence,
    and no team evidence on most frames without a ball."""
    rng = np.random.default_rng(seed)
    n = match.n_frames
    kernel = np.ones(40) / 40
    stopped = np.arange(n) % 1000 >= 800
    noise = np.convolve(rng.normal(0.0, 8.0, n), kernel, mode="same")
    alive = 1 / (1 + np.exp(-(np.where(stopped, -3.0, 2.0) + noise)))
    home = 1 / (1 + np.exp(-np.convolve(rng.normal(0.0, 12.0, n), kernel, mode="same")))
    home[~match.ball_seen & (rng.random(n) < 0.7)] = np.nan
    return alive, home


@pytest.mark.parametrize("params", [{}, V1, V1_SNAP], ids=["defaults", "v1", "v1_snap"])
def test_default_output_equals_the_previous_implementation(params):
    match = make_match(n=6000, seed=3)
    match.half[3000:] = 2
    p_alive, p_home = _evidence(match)
    before = _previous_switch_cost(match, p_alive, p_home, **params)
    after = switch_cost(match, p_alive, p_home, **params)
    np.testing.assert_array_equal(after, before)
    assert len(set(before.tolist())) == 4  # the synthetic case exercises H, A, D and X
    off = {
        "missing_frames_per_observation": 5.0,
        "line_snap_s": 0.0,
        "reentry_snap_s": 0.0,
        "short_spell_s": 2.0,
        "short_spell_policy": "keep",
        "change_shift_frames": 0,
        "min_stop_s": 0.0,
        "min_stop_min_alive": 0.5,
    }
    np.testing.assert_array_equal(switch_cost(match, p_alive, p_home, **params, **off), before)


def test_every_option_gives_valid_labels_and_rejects_an_unknown_policy():
    match = make_match(n=4000, seed=5)
    p_alive, p_home = _evidence(match, seed=1)
    options = {
        "missing_frames_per_observation": 25.0,
        "line_snap_s": 4.0,
        "reentry_snap_s": 4.0,
        "short_spell_s": 3.0,
        "short_spell_policy": "dead",
        "short_spell_max_llr": 2.2,
        "change_shift_frames": 4,
        "min_stop_s": 4.0,
        "min_stop_min_alive": 0.2,
    }
    labels = switch_cost(match, p_alive, p_home, **V1_SNAP, **options)
    assert labels.shape == (match.n_frames,) and set(labels.tolist()) <= {"H", "A", "D", "X"}
    assert (labels[~match.graded] == "X").all()
    with pytest.raises(ValueError, match="short_spell_policy"):
        switch_cost(match, p_alive, p_home, short_spell_policy="drop")


# -- small hand-made matches -------------------------------------------------------------------


def _match(ball_xy, half=None):
    """One graded half; two still players far from the ball; ball (n, 2), NaN = not tracked."""
    n = len(ball_xy)
    players = np.tile(np.array([[30.0, 30.0, 0.0], [70.0, 30.0, 0.0]]), (n, 1, 1))
    return Match(
        game="SYNTH", fps=FPS, pitch=(105.0, 68.0),
        ball=np.concatenate([ball_xy, np.full((n, 1), np.nan)], axis=1), players=players,
        team_of_track=np.array([HOME, AWAY], dtype=np.int8),
        half=np.ones(n, dtype=np.int8) if half is None else half,
    )  # fmt: skip


def _labels(n, *spans):
    """H everywhere, then each (start, stop, letter) painted in order."""
    labels = np.full(n, "H", dtype="<U1")
    for start, stop, letter in spans:
        labels[start:stop] = letter
    return labels


def _across_touchline(n=600, out_at=200):
    """A ball rolling at 5 m/s towards the touchline y = 0, beyond it from frame `out_at`."""
    t = np.arange(n, dtype=float)
    y = 0.2 * (out_at - t) - 0.6  # 0.6 m beyond the line at `out_at`, more after
    return np.stack([np.full(n, 50.0), np.clip(y, -3.0, 30.0)], axis=1)


def test_line_exit_starts_the_stoppage_where_the_ball_left():
    ball = _across_touchline()
    labels = _labels(600, (250, 400, "D"))  # decoded 2 s after the ball left the pitch
    match = _match(ball)
    dead = np.flatnonzero(snap_to_line_exit(match, labels, within_s=4.0, margin_m=0.5) == "D")
    assert dead[0] == 200 and dead[-1] == 399
    unchanged = snap_to_line_exit(match, labels, within_s=1.0, margin_m=0.5)
    np.testing.assert_array_equal(unchanged, labels)  # the exit is further back than 1 s
    back = ball.copy()
    back[230:250] = [50.0, 1.0]  # seen back inside the pitch in between: not this exit
    np.testing.assert_array_equal(snap_to_line_exit(_match(back), labels, 4.0, 0.5), labels)
    again = ball.copy()
    again[230:235] = [50.0, 1.0]  # back inside, then out again at 235: that exit counts
    assert np.flatnonzero(snap_to_line_exit(_match(again), labels, 4.0, 0.5) == "D")[0] == 235
    gap = ball.copy()
    gap[210:300] = np.nan  # lost after leaving: the exit still counts
    assert np.flatnonzero(snap_to_line_exit(_match(gap), labels, 4.0, 0.5) == "D")[0] == 200
    shallow = ball.copy()
    shallow[200:] = [50.0, -0.3]  # only 0.3 m beyond the line: inside the margin
    np.testing.assert_array_equal(snap_to_line_exit(_match(shallow), labels, 4.0, 0.5), labels)
    other_half = np.ones(600, dtype=np.int8)
    other_half[220:] = 2  # the exit and the stoppage lie in different halves
    np.testing.assert_array_equal(
        snap_to_line_exit(_match(ball, other_half), labels, 4.0, 0.5), labels
    )


def _throw_in(n=600, throw_at=300):
    """The ball is held 1 m beyond the touchline, then thrown in at 10 m/s at `throw_at`."""
    t = np.arange(n, dtype=float)
    y = np.where(t < throw_at, -1.0, -1.0 + 0.4 * (t - throw_at + 1))
    return np.stack([np.full(n, 50.0), np.minimum(y, 40.0)], axis=1)


def test_reentry_moves_the_restart_onto_the_throw():
    ball = _throw_in()
    match = _match(ball)
    entry = 302  # first frame on or inside the line: y = -1.0 + 0.4 * 3 >= 0
    late = _labels(600, (100, 340, "D"), (340, 600, "A"))
    moved = snap_to_reentry(match, late, within_s=2.0, margin_m=0.5, restart_speed_m_s=3.0)
    assert np.flatnonzero(moved == "D")[-1] == entry - 1 and moved[entry] == "A"
    early = _labels(600, (100, 280, "D"), (280, 600, "A"))
    moved = snap_to_reentry(match, early, within_s=2.0, margin_m=0.5, restart_speed_m_s=3.0)
    assert np.flatnonzero(moved == "D")[-1] == entry - 1 and moved[entry] == "A"
    far = _labels(600, (100, 400, "D"), (400, 600, "A"))  # 4 s after the throw: left alone
    np.testing.assert_array_equal(snap_to_reentry(match, far, 2.0, 0.5, 3.0), far)
    slow = ball.copy()
    slow[300:, 1] = np.minimum(-1.0 + 0.04 * (np.arange(300, 600) - 299), 40.0)  # carried in
    np.testing.assert_array_equal(snap_to_reentry(_match(slow), late, 6.0, 0.5, 3.0), late)


def test_short_spells_between_two_of_the_other_team():
    n = 600
    labels = _labels(n, (200, 230, "A"))  # 1.2 s of the away team inside a home possession
    match = _match(np.full((n, 2), 50.0))
    home = np.full(n, 0.9)
    home[200:230] = 0.2  # |log-odds| 1.39 on the spell
    dead = short_spells(match, labels, home, max_s=2.0, policy="dead")
    assert (dead[200:230] == "D").all() and (dead[:200] == "H").all() and (dead[230:] == "H").all()
    merged = short_spells(match, labels, home, max_s=2.0, policy="merge")
    assert (merged == "H").all()
    np.testing.assert_array_equal(short_spells(match, labels, home, 1.0, "dead"), labels)  # long
    np.testing.assert_array_equal(short_spells(match, labels, home, 2.0, "keep"), labels)
    np.testing.assert_array_equal(short_spells(match, labels, home, 2.0, "dead", 1.0), labels)
    assert (short_spells(match, labels, home, 2.0, "dead", 2.2)[200:230] == "D").all()
    unknown = home.copy()
    unknown[200:230] = np.nan  # no team evidence on the spell counts as no confidence
    assert (short_spells(match, labels, unknown, 2.0, "merge", 0.5) == "H").all()
    beside_dead = _labels(n, (200, 230, "A"), (230, 300, "D"))
    np.testing.assert_array_equal(short_spells(match, beside_dead, home, 2.0, "dead"), beside_dead)
    chain = _labels(n, (200, 220, "A"), (220, 230, "H"), (230, 600, "A"))  # 0.8 s A, 0.4 s H
    resolved = short_spells(match, chain, home, 2.0, "merge")
    assert (resolved[:200] == "H").all() and (resolved[200:] == "A").all()  # shortest first


def test_short_stoppages_inside_one_teams_possession_become_that_team():
    n = 600
    match = _match(np.full((n, 2), 50.0))
    alive = np.full(n, 0.9)
    alive[200:250] = 0.1  # 2 s stoppage, weak-ish dead evidence (mean 0.1)
    inside = _labels(n, (200, 250, "D"))  # H, 2 s D, H
    assert (short_stoppages(match, inside, alive, 3.0) == "H").all()
    np.testing.assert_array_equal(short_stoppages(match, inside, alive, 2.0), inside)  # not shorter
    np.testing.assert_array_equal(short_stoppages(match, inside, alive, 0.0), inside)  # off
    np.testing.assert_array_equal(short_stoppages(match, inside, alive, 3.0, 0.2), inside)
    assert (short_stoppages(match, inside, alive, 3.0, 0.09) == "H").all()  # mean 0.1 >= 0.09
    away = _labels(n, (200, 250, "D"), (250, 600, "A"))  # H, D, A: a real hand-over stays
    np.testing.assert_array_equal(short_stoppages(match, away, alive, 3.0), away)
    edge = _labels(n, (0, 50, "D"), (560, 600, "D"))  # at the edges of the half
    np.testing.assert_array_equal(short_stoppages(match, edge, alive, 3.0), edge)
    half = np.ones(n, dtype=np.int8)
    half[250:] = 2  # neighbours in different halves
    np.testing.assert_array_equal(
        short_stoppages(_match(np.full((n, 2), 50.0), half), inside, alive, 3.0), inside
    )
    beside_x = _labels(n, (200, 250, "D"), (250, 260, "X"))
    np.testing.assert_array_equal(short_stoppages(match, beside_x, alive, 3.0), beside_x)
    two = _labels(n, (100, 140, "D"), (300, 400, "D"))  # 1.6 s goes, 4 s stays
    out = short_stoppages(match, two, alive, 3.0)
    assert (out[100:140] == "H").all() and (out[300:400] == "D").all()


def test_shift_moves_direct_changes_earlier_only():
    labels = _labels(300, (100, 200, "A"), (200, 250, "D"), (250, 300, "A"))
    shifted = shift_changes(labels, 4)
    assert np.flatnonzero(shifted == "A")[0] == 96  # H -> A moved 4 frames earlier
    assert np.flatnonzero(shifted == "D")[0] == 200 and np.flatnonzero(shifted == "D")[-1] == 249
    short = _labels(300, (100, 103, "A"), (103, 300, "H"))  # a 3-frame spell keeps one frame
    shifted = shift_changes(short, 6)  # of the spell before its end, so its end moves only 2
    assert np.flatnonzero(shifted == "A").tolist() == list(range(94, 101))
    np.testing.assert_array_equal(shift_changes(labels, 0), labels)


def test_missing_ball_evidence_counts_less_with_its_own_weight():
    n = 600
    ball = np.full((n, 2), 50.0)
    ball[200:260] = np.nan  # 2.4 s without a ball, during which the team evidence says away
    match = _match(ball)
    alive = np.full(n, 0.95)
    home = np.full(n, 0.9)
    home[200:260] = 0.02
    by_default = switch_cost(match, alive, home, **V1)
    assert (by_default[200:260] == "A").all()  # 60 frames x 3.9 nats / 5 outweigh two changes
    weighed = switch_cost(match, alive, home, **V1, missing_frames_per_observation=25.0)
    assert (weighed == "H").all()  # 60 x 3.9 / 25 = 9.3 nats < 2 x 5
