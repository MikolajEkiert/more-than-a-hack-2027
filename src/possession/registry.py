"""The method registry: the mechanism that keeps three people and their agents from writing the
same thing three times.

Every algorithm is a function registered for exactly one stage:

    @register("team", "nearest_player")
    def nearest_player(match, max_distance_m=3.0):
        '''One line saying what it does.'''
        ...

Rules enforced here (and by tests/test_contracts.py for every registered method):
  * a stage is one of STAGES; a name is unique within its stage
  * the function has a docstring, because `possession methods` prints its first line
  * every parameter after the positional inputs has a default, so the method runs from a config
    that names it and nothing else
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

STAGES = ("ball_state", "team", "decode")

# How many positional inputs each stage receives before its keyword parameters.
#   ball_state(match, **params)                -> p_alive  float32 (n,) in [0, 1], no NaN
#   team(match, **params)                      -> p_home   float32 (n,) in [0, 1], NaN = no evidence
#   decode(match, p_alive, p_home, **params)   -> labels   <U1 (n,) in {"H", "A", "D", "X"}
POSITIONAL = {"ball_state": 1, "team": 1, "decode": 3}


@dataclass(frozen=True)
class Method:
    stage: str
    name: str
    fn: Callable
    summary: str
    params: dict  # name -> default


_REGISTRY: dict[str, dict[str, Method]] = {stage: {} for stage in STAGES}


def register(stage: str, name: str):
    """Decorator registering `fn` as method `name` of `stage`."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")

    def decorator(fn):
        if name in _REGISTRY[stage]:
            existing = _REGISTRY[stage][name].fn
            raise ValueError(
                f"{stage}.{name} is already registered by {existing.__module__}."
                f"{existing.__name__}; extend it or pick another name"
            )
        doc = inspect.getdoc(fn)
        if not doc:
            raise ValueError(f"{stage}.{name} needs a docstring")

        params = list(inspect.signature(fn).parameters.values())
        n_pos = POSITIONAL[stage]
        if len(params) < n_pos:
            raise TypeError(f"{stage}.{name} must take {n_pos} positional input(s)")
        keyword = params[n_pos:]
        missing = [p.name for p in keyword if p.default is inspect.Parameter.empty]
        if missing:
            raise TypeError(f"{stage}.{name}: parameters {missing} need defaults")

        _REGISTRY[stage][name] = Method(
            stage=stage,
            name=name,
            fn=fn,
            summary=doc.splitlines()[0],
            params={p.name: p.default for p in keyword},
        )
        return fn

    return decorator


def get(stage: str, name: str) -> Method:
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; stages are {STAGES}")
    try:
        return _REGISTRY[stage][name]
    except KeyError:
        have = ", ".join(sorted(_REGISTRY[stage])) or "none"
        raise KeyError(f"no {stage} method {name!r}; registered: {have}") from None


def methods(stage: str | None = None) -> list[Method]:
    stages = (stage,) if stage else STAGES
    return [m for s in stages for _, m in sorted(_REGISTRY[s].items())]
