# Architecture

```
data/ ──loader──> Match ──ball_state──> p_alive ──team──> p_home ──decode──> labels ──> CSV
         (cached)          (method A)               (method B)            (method C)
                       config TOML picks A, B, C and their parameters
```

## Why three stages

The label space factorises: `D` vs alive is one question (stoppages, restarts, ball out), and
`H` vs `A` *given alive* is another (control, hand-overs). Structure over time (possessions
as intervals, which is what spellF1 scores) is a third. Splitting them lets three people work in
parallel on separate files, compare methods for one stage while the other two stay fixed, and
reason about which failure mode a change attacks.

## Contracts (enforced by `pipeline.py` and `tests/test_contracts.py`)

| stage | signature | returns |
|---|---|---|
| `ball_state` | `fn(match, **params)` | `p_alive` float (n,) in [0, 1], no NaN |
| `team` | `fn(match, **params)` | `p_home` float (n,) in [0, 1], **NaN = no evidence** |
| `decode` | `fn(match, p_alive, p_home, **params)` | labels `<U1` (n,) in {H, A, D, X} |

- `n = match.n_frames` = 199,319 for BAR-ATH; index = video frame.
- The pipeline then sets every ungraded frame to `X`.
- Every keyword parameter has a default; configs override them; unknown keys are errors.
- Methods are pure and deterministic and never mutate `Match`.
- Soft outputs are deliberate: a decoder (HMM, hysteresis, minimum durations) can use confidence.
  A method with only a hard answer returns 0.0 or 1.0.

If a stage needs more than this (e.g. decode wants a per-frame confidence of the team signal), change
the contract **in one small PR** that updates this table, `pipeline.py`, the contract tests and
docs/DECISIONS.md, before building on it.

## `Match` (src/possession/match.py)

| field | shape | meaning |
|---|---|---|
| `ball` | (n, 3) | x, y in metres, speed m/s; NaN where not tracked (41% of graded frames) |
| `players` | (n, S, 3) | same per track slot; S = 46 slots (tracklets, **not** people) |
| `team_of_track` | (S,) | 0 home, 1 away |
| `half` | (n,) | 1 or 2 inside a graded half, 0 outside |
| `tagging` | list or None | Smart Tagging instances, with `start_frame` and `end_frame` in video frames |
| `pitch`, `fps`, `game` | | constants from `games.py` |

Derived quantities shared by several methods live in `features.py` and are memoised per match
(`features.nearest_player`, `features.tagging_mask`, ...).

## Evaluation without tuning on the public window

- `scoring.diagnostics`: possession count, mean duration, home share and label shares, all with no GT.
- `scoring.public_sanity`: the public 5 minutes; a bug detector only.
- Other games with labels derived from their own data are legitimate validation (RULES.md),
  see tasks/02.

## Reproducibility

`possession submit` runs the pipeline, re-parses the data from scratch, runs again, requires an
identical sha256, and calls the official validator. The CSV is written with explicit `\n` so the
bytes do not depend on the OS. `uv.lock` pins every dependency; `.python-version` pins 3.11.
