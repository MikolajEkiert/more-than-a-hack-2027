# src/possession/ - rules for writing pipeline code

Scoped addendum to the root AGENTS.md (read that first: goal, hard rules, workflow). Claude Code
loads this via `CLAUDE.md` (`@AGENTS.md`); Cursor reads it natively. Detail: docs/ARCHITECTURE.md.

## Before you write
- `uv run possession methods` first. Extend or parameterise an existing method before adding a copy.
- Look in `features.py`, `starter/` and `evaluation/possession_metrics.py` for the helper you need.

## Methods
- One registered function per idea, in `stages/<stage>.py`: `@register(stage, name)`. Docstring
  first line = what it does (shown by `possession methods`). A default for every parameter.
  Never a second pipeline: a new idea is a new method picked by a config.
- Contract (enforced by `pipeline.py` and `tests/test_contracts.py`):

| stage | signature | returns |
|---|---|---|
| `ball_state` | `fn(match, **params)` | `p_alive` float (n,) in [0, 1], no NaN |
| `team` | `fn(match, **params)` | `p_home` float (n,) in [0, 1], NaN = no evidence |
| `decode` | `fn(match, p_alive, p_home, **params)` | labels `<U1` (n,) in {H, A, D, X} |

- Quantities several methods need go in `features.py`, memoised via `_memo`, not inline.
- Vectorised numpy: n = 199,319 frames, no per-frame Python loops in methods.
- NaN-safe: the ball is missing on 41% of graded frames, players drop out.
- Deterministic: randomness only through an explicit `seed` parameter; no set/dict-order
  dependence. Never mutate `Match`. No prints (CLI only).
- Index = video frame (ATD frame_id - 1). Distances in metres (`Match` is already scaled).
- No constants tied to frames or moments of this match (RULES.md 1).

## Shared contracts
`match.py`, `registry.py`, `pipeline.py`, `config.py`: change only in a small separate PR that
also updates docs/ARCHITECTURE.md, the contract tests and one line in docs/DECISIONS.md.

## Tests and done
- Add a focused test for each new method, using `make_match` from `tests/conftest.py`. The
  contract tests cover every registered method automatically.
- `make check` green = done. Organiser files (`evaluation/`, `starter/`) are read-only: import them.
