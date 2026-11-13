"""Experiment configs: one TOML file in configs/ fully describes one submission.

    name = "baseline_smooth50"      # becomes evaluation/results/<name>.csv
    game = "BAR-ATH"                # optional, defaults to games.GAME

    [ball_state]
    method = "ball_visible"

    [team]
    method = "nearest_player"

    [decode]
    method = "majority_vote"
    window = 50                     # any other key is a keyword parameter of the method

Unknown parameters are an error, not a silent no-op: a typo in a config must not produce a run
that looks like it tested something it did not.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from possession import registry


@dataclass(frozen=True)
class StageConfig:
    method: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    name: str
    stages: dict  # stage -> StageConfig
    game: str | None = None
    path: str | None = None

    def as_dict(self) -> dict:
        out = {"name": self.name, "game": self.game}
        for stage, sc in self.stages.items():
            out[stage] = {"method": sc.method, **sc.params}
        return out


def parse(data: dict, path: str | None = None) -> Config:
    data = dict(data)
    name = data.pop("name", None) or (Path(path).stem if path else None)
    if not name:
        raise ValueError("config needs a name")
    game = data.pop("game", None)

    stages = {}
    for stage in registry.STAGES:
        section = dict(data.pop(stage, {}) or {})
        if "method" not in section:
            raise ValueError(f"{path or name}: [{stage}] needs a 'method'")
        method = registry.get(stage, section.pop("method"))
        unknown = sorted(set(section) - set(method.params))
        if unknown:
            raise ValueError(
                f"{path or name}: {stage}.{method.name} has no parameter(s) {unknown}; "
                f"it takes {sorted(method.params)}"
            )
        stages[stage] = StageConfig(method.name, section)
    if data:
        raise ValueError(f"{path or name}: unknown top-level keys {sorted(data)}")
    return Config(name=name, stages=stages, game=game, path=path)


def load(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as handle:
        return parse(tomllib.load(handle), str(path))
