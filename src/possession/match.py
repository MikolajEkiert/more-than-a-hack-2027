"""`Match`: the single in-memory representation of one game that every method receives.

Methods read from it and never mutate it. If a method needs a quantity derived from the raw
arrays that several methods would share (ball speed, team centroids, ...), add it to
`possession/features.py` as a cached function of the match rather than recomputing it inline.
"""

from dataclasses import dataclass, field

import numpy as np

HOME, AWAY = 0, 1


@dataclass(frozen=True, eq=False)
class Match:
    game: str
    fps: int
    pitch: tuple[float, float]  # (length, width) in metres
    ball: np.ndarray  # (n, 3) float64: x_m, y_m, speed_m_s; NaN where not tracked
    players: np.ndarray  # (n, S, 3) float64: x_m, y_m, speed_m_s per track slot; NaN if absent
    team_of_track: np.ndarray  # (S,) int8: HOME or AWAY for each slot
    half: np.ndarray  # (n,) int8: 1 or 2 inside a graded half, 0 outside (boundaries, not labels)
    # Smart Tagging, [{code, start_frame, end_frame, start_s, end_s, tags}], or None if absent.
    tagging: list | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])

    @property
    def graded(self) -> np.ndarray:
        """Frames inside a half. Ungraded frames cost nothing whatever is written there."""
        return self.half > 0

    @property
    def ball_seen(self) -> np.ndarray:
        return ~np.isnan(self.ball[:, 0])

    def validate(self) -> None:
        n, s = self.n_frames, self.team_of_track.shape[0]
        if self.ball.shape != (n, 3):
            raise ValueError(f"ball shape {self.ball.shape} != ({n}, 3)")
        if self.players.shape != (n, s, 3):
            raise ValueError(f"players shape {self.players.shape} != ({n}, {s}, 3)")
        if self.half.shape != (n,):
            raise ValueError(f"half shape {self.half.shape} != ({n},)")
        if not np.isin(self.team_of_track, (HOME, AWAY)).all():
            raise ValueError("team_of_track must be HOME (0) or AWAY (1)")
