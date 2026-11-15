"""The optional degradation mechanisms of task 14, and that they leave every existing profile as it
was: the feature cache and the trained models are keyed on the degraded matches and on the
profile dicts (regime.py), so a changed byte would silently invalidate hours of computation."""

import json
from dataclasses import replace

import numpy as np
import pytest

from possession import validation
from possession.features import runs
from possession.match import AWAY, HOME, Match

EXISTING = ("label_free", "match_like", "published_baselines")


def _reference_degrade(match, labels, seed, profile):
    """`validation.degrade` as it was before task 14 (commit 9aaf4af), frozen as the oracle."""
    p = validation.degradation_profile(profile)
    rng = np.random.default_rng(seed)
    n, fps = match.n_frames, match.fps
    ball = match.ball.copy()
    players = match.players.copy()
    alive = np.isin(labels, ("H", "A"))
    dead_starts, _ = runs(labels == "D")
    for start in dead_starts:
        if start == 0 or not alive[start - 1] or rng.random() >= p["visible_dead_prob"]:
            continue
        length = int(rng.uniform(*p["visible_dead_s"]) * fps)
        last = ball[start - 1, :2]
        back = max(start - 6, 0)
        velocity = (last - ball[back, :2]) / max(start - 1 - back, 1)
        if np.isnan(velocity).any() or np.isnan(last).any():
            continue
        stop = min(start + length, n)
        steps = np.arange(1, stop - start + 1)[:, None]
        ball[start:stop, :2] = last + velocity * steps
    if p["ball_offset_m"] > 0:
        window = fps
        noise = rng.normal(size=(n + window, 2))
        smooth = np.stack(
            [np.convolve(noise[:, a], np.ones(window) / window, mode="valid")[:n] for a in (0, 1)],
            axis=1,
        )
        ball[:, :2] += smooth * p["ball_offset_m"] * np.sqrt(window)
    window = int(round(p["ball_smooth_s"] * fps))
    if window > 1:
        tracked = ~np.isnan(ball[:, 0])
        kernel = np.ones(window)
        count = np.convolve(tracked.astype(np.float64), kernel, mode="same")
        for axis in (0, 1):
            total = np.convolve(np.where(tracked, ball[:, axis], 0.0), kernel, mode="same")
            ball[:, axis] = np.where(tracked, total / np.maximum(count, 1.0), np.nan)
    hidden = np.zeros(n, dtype=bool)
    target = p["lost_ball_share"] * alive.sum()
    for index in rng.permutation(len(dead_starts)):
        if (hidden & alive).sum() >= target / 2:
            break
        start = dead_starts[index]
        hidden[max(start - int(rng.uniform(*p["lead_in_s"]) * fps), 0) : start] = True
    alive_frames = np.flatnonzero(alive)
    for _ in range(10 * len(dead_starts) + 100):
        if (hidden & alive).sum() >= target or not len(alive_frames):
            break
        start = int(rng.choice(alive_frames))
        hidden[start : start + int(rng.uniform(*p["free_gap_s"]) * fps)] = True
    ball[hidden] = np.nan
    visible = ~np.isnan(players[:, :, 0])
    mean_visible = visible[match.graded].sum(axis=1).mean() if match.graded.any() else 0.0
    if mean_visible > p["players_visible"]:
        drop = 1.0 - p["players_visible"] / mean_visible
        mean_out = 20.0 * fps
        mean_in = mean_out * (1.0 - drop) / drop
        for slot in range(players.shape[1]):
            frame = int(rng.exponential(mean_in))
            while frame < n:
                out = int(rng.exponential(mean_out)) + 1
                players[frame : frame + out, slot] = np.nan
                frame += out + int(rng.exponential(mean_in)) + 1
    return replace(match, game=f"{match.game}@{profile}#{seed}", ball=ball, players=players)


def _match_with_stoppages(n=9000, slots=22, fps=25, seed=1):
    """A clean external match: the ball follows a drifting player and is missing exactly during
    the stoppages (as in corrected tracking), the last one running into the end of the half."""
    rng = np.random.default_rng(seed)
    xy = np.cumsum(rng.normal(0, 0.1, (n, slots, 2)), axis=0) + rng.uniform(20, 50, (1, slots, 2))
    players = np.concatenate([xy, validation.speeds(xy, fps)[..., None]], axis=2)
    ball = np.concatenate([xy[:, 0] + 0.3, np.full((n, 1), np.nan)], axis=1)
    labels = np.full(n, "H", dtype="<U1")
    labels[4500:] = "A"
    for start, length in ((0, 100), (1000, 300), (2500, 200), (4400, 400), (6000, 250)):
        labels[start : start + length] = "D"
    labels[8800:] = "D"
    half = np.ones(n, dtype=np.int8)
    half[8900:] = 0
    labels[half == 0] = "X"
    ball[labels == "D"] = np.nan
    match = Match(
        game="SYNTH", fps=fps, pitch=(105.0, 68.0), ball=ball, players=players,
        team_of_track=np.array([HOME, AWAY] * (slots // 2), dtype=np.int8), half=half,
    )  # fmt: skip
    return match, labels


@pytest.mark.parametrize("profile", EXISTING)
def test_existing_profiles_degrade_byte_for_byte_as_before(profile):
    match, labels = _match_with_stoppages()
    for seed in (0, 1, 2):
        new = validation.degrade(match, labels, seed=seed, profile=profile)
        old = _reference_degrade(match, labels, seed, profile)
        assert new.ball.tobytes() == old.ball.tobytes(), (profile, seed)
        assert new.players.tobytes() == old.players.tobytes(), (profile, seed)
        assert new.game == old.game


def test_profile_dicts_are_the_files_params_exactly():
    for name in validation.degradation_profiles():
        path = validation.PROFILE_DIR / f"{name}.json"
        assert validation.degradation_profile(name) == json.loads(path.read_text())["params"]
    for name in EXISTING:  # no optional key was added to an existing profile
        assert set(validation.degradation_profile(name)) == set(validation.DEGRADATION_KEYS)


def test_neutral_values_of_the_optional_keys_change_nothing():
    match, labels = _match_with_stoppages()
    base = validation.degrade(match, labels, seed=4, profile="match_like")
    same = validation.degrade(
        match, labels, seed=4, profile="match_like", **validation.OPTIONAL_DEGRADATION_KEYS
    )
    assert base.ball.tobytes() == same.ball.tobytes()
    assert base.players.tobytes() == same.players.tobytes()
    with pytest.raises(TypeError):
        validation.degrade(match, labels, placed_ball_probability=1.0)  # a typo is an error


def _restarts(labels):
    starts, stops = runs(labels == "D")
    alive = np.isin(labels, ("H", "A"))
    return [(a, b) for a, b in zip(starts, stops, strict=True) if b < len(labels) and alive[b]]


QUIET = dict(lost_ball_share=0.0, visible_dead_prob=0.0, ball_offset_m=0.0, ball_smooth_s=0.0,
             players_visible=99.0)  # fmt: skip


def test_placed_ball_lies_still_at_the_restart_spot():
    match, labels = _match_with_stoppages()
    out = validation.degrade(
        match, labels, seed=0, placed_ball_prob=1.0, placed_ball_s=[2.0, 2.0], **QUIET
    )
    restarts = _restarts(labels)
    assert len(restarts) == 5
    for start, stop in restarts:
        still = max(start, stop - 50)
        spot = match.ball[stop, :2]
        np.testing.assert_array_equal(
            out.ball[still:stop, :2], np.broadcast_to(spot, (stop - still, 2))
        )
        assert np.isnan(out.ball[start:still, 0]).all()  # nothing carried: lost before that
    # play is untouched, and so is the end of the half (no restart)
    live = np.isin(labels, ("H", "A"))
    np.testing.assert_array_equal(out.ball[live], match.ball[live])
    assert np.isnan(out.ball[8800:8900, 0]).all()


def test_carried_ball_moves_from_where_it_was_to_the_spot():
    match, labels = _match_with_stoppages()
    out = validation.degrade(
        match, labels, seed=0, placed_ball_prob=1.0, placed_ball_s=[1.0, 1.0],
        carried_ball_s=[3.0, 3.0], **QUIET,
    )  # fmt: skip
    start, stop = 1000, 1300
    still, carry = stop - 25, stop - 25 - 75
    path = out.ball[carry:still, :2]
    origin, spot = match.ball[start - 1, :2], match.ball[stop, :2]
    steps = np.diff(path, axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(steps[0], steps.shape), atol=1e-9)
    direction = (spot - origin) / np.linalg.norm(spot - origin)
    np.testing.assert_allclose(steps[0] / np.linalg.norm(steps[0]), direction, atol=1e-9)
    assert np.isnan(out.ball[start:carry, 0]).all()
    # the first stoppage starts the match: no position before it, the ball is carried in place
    assert np.isfinite(out.ball[:100, 0]).sum() == 100


def test_jitter_is_small_noise_around_the_spot():
    match, labels = _match_with_stoppages()
    out = validation.degrade(
        match, labels, seed=0, placed_ball_prob=1.0, placed_ball_s=[4.0, 4.0],
        placed_jitter_m=0.05, **QUIET,
    )  # fmt: skip
    a, b = 2600, 2700
    offsets = out.ball[a:b, :2] - match.ball[b, :2]
    assert 0.03 < offsets.std() < 0.07 and abs(offsets.mean()) < 0.02


def test_ball_picked_up_late_after_restarts():
    match, labels = _match_with_stoppages()
    out = validation.degrade(match, labels, seed=0, reacquire_s=[1.0, 1.0], **QUIET)
    for _, stop in _restarts(labels):
        assert np.isnan(out.ball[stop : stop + 25, 0]).all()
        assert not np.isnan(out.ball[stop + 25 : stop + 50, 0]).any()
    lost = np.isnan(out.ball[:, 0]) & ~np.isnan(match.ball[:, 0])
    assert lost.sum() == 25 * len(_restarts(labels))


def test_lead_in_fraction_puts_the_whole_loss_before_stoppages():
    match, labels = _match_with_stoppages()
    out = validation.degrade(
        match, labels, seed=2, lost_ball_share=0.02, lead_in_fraction=1.0,
        lead_in_s=[2.0, 4.0], free_gap_s=[5.0, 5.0], visible_dead_prob=0.0, ball_offset_m=0.0,
        ball_smooth_s=0.0, players_visible=99.0,
    )  # fmt: skip
    lost = np.isnan(out.ball[:, 0]) & np.isin(labels, ("H", "A"))
    dead = labels == "D"
    for _, b in zip(*runs(lost), strict=True):
        assert dead[b], "a stretch of play without the ball that does not run into a stoppage"


def test_new_mechanisms_use_their_own_random_stream():
    """Switching the placed ball on changes the ball in stoppages only: the loss in play, the
    offset and the players come from the first stream, as before."""
    match, labels = _match_with_stoppages()
    base = validation.degrade(match, labels, seed=5, profile="match_like")
    placed = validation.degrade(
        match, labels, seed=5, profile="match_like", placed_ball_prob=0.7,
        placed_ball_s=[1.0, 3.0], carried_ball_s=[0.0, 2.0],
    )  # fmt: skip
    live = np.isin(labels, ("H", "A"))
    assert base.ball[live].tobytes() == placed.ball[live].tobytes()
    assert base.players.tobytes() == placed.players.tobytes()
    assert (
        np.isfinite(placed.ball[labels == "D", 0]).sum()
        > np.isfinite(base.ball[labels == "D", 0]).sum()
    )


def test_calibrate_to_match_takes_points_a_vector_and_a_loss(monkeypatch):
    calls = []

    def fake_job(job):
        _, _, game, seed, _, overrides, vector = job
        calls.append((game, seed, vector))
        share = overrides["lost_ball_share"]
        out = {"possession_agreement": share, "team_agreement": 0.8, "team_margin": 0.7,
               "ball_missing": 0.4}  # fmt: skip
        if vector:
            out["missing_given_live"] = share / 2 if game == "g1" else None
        return out

    monkeypatch.setattr(validation, "_agreement_job", fake_job)
    target = {"possession_agreement": 0.3, "team_agreement": 0.8, "team_margin": 0.7}
    seen = []
    rows = validation.calibrate_to_match(
        "a", "b", target, games=("g1", "g2"), points=[{"lost_ball_share": 0.1},
        {"lost_ball_share": 0.25}], vector=True, progress=seen.append,
        loss=lambda stats, t: abs(stats["possession_agreement"] - t["possession_agreement"]),
    )  # fmt: skip
    assert [r["overrides"]["lost_ball_share"] for r in rows] == [0.25, 0.1]  # best first
    assert [r["overrides"]["lost_ball_share"] for r in seen] == [0.1, 0.25]  # as they finish
    assert rows[0]["missing_given_live"] == 0.125  # a statistic one match lacks is not a zero
    assert rows[0]["loss"] == 0.05 and len(calls) == 4 and all(c[2] for c in calls)

    calls.clear()
    default = validation.calibrate_to_match("a", "b", target, games=("g1",))
    assert len(default) == len(calls) == 32 and not any(c[2] for c in calls)
    expected = validation.agreement_loss(default[0], target)
    assert default[0]["loss"] == pytest.approx(expected, abs=1e-4)
    assert "missing_given_live" not in default[0]


# -- task 15: a tracked ball lying still while play goes on --------------------------------------

MATCH_REGIME_GOLDEN = "3590fce405c6ed7f8c0e2b1d7f8951fcab9c56fc243a62461716ad2ab9ba4c75"


def test_match_regime_degrades_exactly_as_before_task_15():
    """The profile fitted in task 14 must not move: its degraded matches key cached features."""
    import hashlib

    match, labels = _match_with_stoppages()
    out = validation.degrade(match, labels, seed=3, profile="match_regime")
    digest = hashlib.sha256(
        np.ascontiguousarray(out.ball).tobytes() + np.ascontiguousarray(out.players).tobytes()
    ).hexdigest()
    assert digest == MATCH_REGIME_GOLDEN


def _still_runs(ball, alive, fps=25):
    still = np.zeros(len(ball), dtype=bool)
    step = np.hypot(*np.diff(ball[:, :2], axis=0).T) * fps
    still[1:] = step < 1.0
    starts, stops = runs(still & alive)
    return [(a, b) for a, b in zip(starts, stops, strict=True) if b - a >= fps // 2]


def test_still_ball_freezes_the_track_or_follows_a_spare_ball():
    match, labels = _match_with_stoppages()
    alive = np.isin(labels, ("H", "A"))
    quiet = dict(QUIET, placed_jitter_m=0.0, still_ball_s=[1.0, 2.0])
    base = validation.degrade(match, labels, seed=5, profile="match_regime", **quiet)
    frozen = validation.degrade(
        match, labels, seed=5, profile="match_regime", still_ball_rate=60.0, **quiet
    )
    spare = validation.degrade(
        match, labels, seed=5, profile="match_regime",
        still_ball_rate=60.0, still_ball_spare=1.0, **quiet,
    )  # fmt: skip
    assert len(_still_runs(frozen.ball, alive)) > len(_still_runs(base.ball, alive))

    def differs(a, b):
        return ~np.all(np.isclose(a[:, :2], b[:, :2], equal_nan=True), axis=1)

    changed = np.flatnonzero(differs(spare.ball, base.ball) & alive)
    assert len(changed)
    y = spare.ball[changed, 1]
    assert np.all((y < 0) | (y > match.pitch[1]))  # every spare ball lies outside a touch line
    # everything outside the events is untouched, players included
    same = ~differs(frozen.ball, base.ball)
    assert same.mean() > 0.5
    assert frozen.players.tobytes() == base.players.tobytes()
