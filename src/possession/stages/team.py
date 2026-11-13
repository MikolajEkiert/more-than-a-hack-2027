"""Stage 2, team: if the ball is alive, which team has it?

Contract: method(match, **params) -> p_home, float32 (n,), values in [0, 1] or NaN.
1.0 means certainly home, 0.0 certainly away, NaN means this method has no evidence on that frame
(the decoder decides what to do with it). Do not encode "dead ball" here; that is ball_state's job.
"""

import numpy as np

from possession import features
from possession.features import nearest_player as nearest_player_feature
from possession.match import HOME, Match
from possession.registry import register


@register("team", "nearest_player")
def nearest_player(match: Match) -> np.ndarray:
    """Team of the tracked player nearest the ball; NaN where there is no ball or no player."""
    _, team = nearest_player_feature(match)
    out = np.full(match.n_frames, np.nan, dtype=np.float32)
    out[team == HOME] = 1.0
    out[(team >= 0) & (team != HOME)] = 0.0
    return out


@register("team", "control_evidence")
def control_evidence(
    match: Match,
    control_radius_m: float = 2.0,
    margin_scale_m: float = 0.5,
    duel_radius_m: float = 1.0,
    carry_speed_m_s: float = 4.0,
    passing_by_weight: float = 0.3,
) -> np.ndarray:
    """Soft evidence of the team in control of a tracked ball; NaN when nobody is close to it.

    Evidence exists only when a player is within `control_radius_m` of the ball. Its direction
    is the difference between the two teams' nearest players (a logistic of the margin in
    `margin_scale_m`); when both teams are within `duel_radius_m` it is a duel and there is none.
    Its strength depends on whether the ball moves with that player: a ball travelling more than
    `carry_speed_m_s` faster or slower than the nearest player is passing by, not under control,
    and counts `passing_by_weight` as much. Possession between these moments is the decoder's
    job (it keeps the last owner), which is the provider's rule for loose and contested balls.
    """
    near = features.nearest_by_team(match)
    d_home, d_away = near["distance"][:, 0], near["distance"][:, 1]
    with np.errstate(invalid="ignore"):
        closest = np.fmin(d_home, d_away)
        margin = np.where(np.isnan(d_away), np.inf, d_away) - np.where(
            np.isnan(d_home), np.inf, d_home
        )
        has = closest < control_radius_m
        duel = (d_home < duel_radius_m) & (d_away < duel_radius_m)
    lean = 1.0 / (1.0 + np.exp(-np.clip(margin / margin_scale_m, -30, 30)))

    # Speed of the ball relative to the nearest player: small when it is being carried or held.
    ball_speed = features.ball_kinematics(match)["speed"]
    home_closer = np.where(np.isnan(d_away), True, d_home <= d_away)
    slot = np.where(home_closer, near["slot"][:, 0], near["slot"][:, 1])
    player_speed = match.players[np.arange(match.n_frames), np.maximum(slot, 0), 2]
    with np.errstate(invalid="ignore"):
        relative = np.abs(ball_speed - player_speed)
        weight = np.where(relative > carry_speed_m_s, passing_by_weight, 1.0)

    out = np.full(match.n_frames, np.nan, dtype=np.float32)
    ok = has & ~duel & ~np.isnan(lean)
    out[ok] = (0.5 + (lean[ok] - 0.5) * weight[ok]).astype(np.float32)
    return out


@register("team", "touch_episodes")
def touch_episodes(
    match: Match,
    radius_m: float = 2.0,
    duel_radius_m: float = 1.0,
    margin_scale_m: float = 0.5,
    carry_speed_m_s: float = 4.0,
    velocity_change_m_s: float = 6.0,
    min_frames: int = 2,
    touch_p: float = 0.97,
    bridge_p: float = 0.95,
    max_bridge_s: float = 8.0,
    restart_gap_s: float = 2.0,
    restart_radius_m: float = 4.0,
    restart_margin_m: float = 3.0,
    restart_p: float = 0.9,
    restart_hold_s: float = 4.0,
) -> np.ndarray:
    """Team evidence from touches, carried across completed passes; NaN where there is none.

    A touch episode is a run of frames in which one team's player is within `radius_m` of a
    tracked ball and it is not a duel (both teams within `duel_radius_m`). An episode is a real
    touch if the ball moves with the player (its speed within `carry_speed_m_s` of his, for at
    least `min_frames`), or its velocity just after differs from just before by
    `velocity_change_m_s` (the ball was played), or the tracking begins or ends there. A ball
    that flies past a player unchanged is not a touch and gives no evidence.

    Real touches give evidence of strength `touch_p`, leaning by how much nearer that team is
    (`margin_scale_m`). Between two consecutive real touches of the SAME team, with the ball
    tracked all the way and no more than `max_bridge_s` apart, every frame gets evidence
    `bridge_p` for that team: a completed pass or a dribble. Between touches of different teams
    there is none, so the decoder keeps the last owner until the other team's touch.

    Restarts: when the ball is tracked again after at least `restart_gap_s` without it, and one
    team's player is within `restart_radius_m` with the other team's nearest player at least
    `restart_margin_m` further away, that team is taking the restart (opponents must keep their
    distance at a set piece, and the ball usually reappears already in flight, so there is no
    touch to see). It gets evidence `restart_p` from that frame until the next real touch of
    either team, for at most `restart_hold_s`.
    """
    n, fps = match.n_frames, match.fps
    near = features.nearest_by_team(match)
    d_home, d_away = near["distance"][:, 0], near["distance"][:, 1]
    seen = match.ball_seen
    kinematics = features.ball_kinematics(match)
    velocity, ball_speed = kinematics["velocity"], kinematics["speed"]
    with np.errstate(invalid="ignore"):
        closest = np.fmin(d_home, d_away)
        home_closer = np.where(np.isnan(d_away), True, d_home <= d_away)
        duel = (d_home < duel_radius_m) & (d_away < duel_radius_m)
        zone = seen & (closest < radius_m) & ~duel
        margin = np.where(np.isnan(d_away), np.inf, d_away) - np.where(
            np.isnan(d_home), np.inf, d_home
        )
    lean = 1.0 / (1.0 + np.exp(-np.clip(margin / margin_scale_m, -30, 30)))
    slot = np.where(home_closer, near["slot"][:, 0], near["slot"][:, 1])
    player_speed = match.players[np.arange(n), np.maximum(slot, 0), 2]
    with np.errstate(invalid="ignore"):
        relative = np.abs(ball_speed - player_speed)

    # Episodes: maximal runs of in-zone frames with the same nearer team.
    key = np.where(zone, np.where(home_closer, HOME, 1 - HOME), -1)
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    is_episode = key[starts] >= 0
    starts, stops, teams = starts[is_episode], stops[is_episode], key[starts[is_episode]]

    # Ball velocity just before and just after each episode (mean over up to 4 tracked frames).
    def mean_velocity(first, last):
        total = np.zeros((len(first), 2))
        count = np.zeros(len(first))
        for offset in range(4):
            index = first + offset
            ok = (index < last) & (index < n)
            value = velocity[np.minimum(index, n - 1)]
            ok &= ~np.isnan(value[:, 0])
            total[ok] += value[ok]
            count[ok] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / count[:, None]

    before = mean_velocity(np.maximum(starts - 4, 0), starts)
    after = mean_velocity(stops, np.minimum(stops + 4, n))
    changed = np.hypot(*(after - before).T)
    length = stops - starts
    relative_filled = np.where(np.isnan(relative), np.inf, relative)
    carried = np.zeros(len(starts), dtype=bool)
    for i, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist(), strict=True)):
        carried[i] = np.median(relative_filled[start:stop]) < carry_speed_m_s
    with np.errstate(invalid="ignore"):
        real = (length >= min_frames) & (carried | np.isnan(changed))
        real |= changed >= velocity_change_m_s

    out = np.full(n, np.nan, dtype=np.float32)
    strength = 2.0 * touch_p - 1.0
    for start, stop in zip(starts[real].tolist(), stops[real].tolist(), strict=True):
        out[start:stop] = 0.5 + (lean[start:stop] - 0.5) * strength

    # Restarts: who stands at the ball when it is tracked again after a long gap.
    if restart_p > 0.5:
        gap_start, gap_stop = features.runs(~seen)
        long = (gap_stop - gap_start >= restart_gap_s * fps) & (gap_stop < n)
        touch_starts = starts[real]
        hold = int(restart_hold_s * fps)
        for back in gap_stop[long].tolist():
            near_d, far_d = sorted((d_home[back], d_away[back]))
            if np.isnan(near_d) or near_d > restart_radius_m:
                continue
            if not np.isnan(far_d) and far_d - near_d < restart_margin_m:
                continue
            taker_home = bool(home_closer[back])
            # until the next real touch that starts after this frame, the next gap, or the hold
            nxt = int(np.searchsorted(touch_starts, back, side="right"))
            stop = min(back + hold, n)
            if nxt < len(touch_starts):
                stop = min(stop, int(touch_starts[nxt]))
            lost = np.flatnonzero(~seen[back:stop])
            if len(lost):
                stop = back + int(lost[0])
            fill = restart_p if taker_home else 1.0 - restart_p
            segment = out[back:stop]
            out[back:stop] = np.where(np.isnan(segment), fill, segment)

    # Bridge consecutive real touches of the same team.
    r_start, r_stop, r_team = starts[real], stops[real], teams[real]
    max_gap = int(max_bridge_s * fps)
    missing = np.concatenate(([0], np.cumsum(~seen)))
    for i in range(len(r_start) - 1):
        gap_start, gap_stop = int(r_stop[i]), int(r_start[i + 1])
        if r_team[i] != r_team[i + 1] or not 0 < gap_stop - gap_start <= max_gap:
            continue
        if missing[gap_stop] - missing[gap_start]:
            continue  # the ball was lost in between: no claim
        fill = bridge_p if r_team[i] == HOME else 1.0 - bridge_p
        segment = out[gap_start:gap_stop]
        out[gap_start:gap_stop] = np.where(np.isnan(segment), fill, segment)
    return out
