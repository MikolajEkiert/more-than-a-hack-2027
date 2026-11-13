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
- `validation` (`possession validate CONFIG...`): the evidence for choosing between versions,
  within the limits below.
  It loads Metrica's public Sample Game 3 (same provider and EPTS format as the competition
  tracking), derives per-frame H/A/D from it by an algorithm (D where the corrected tracking has
  no ball; otherwise the team of the last on-ball event), degrades the tracking to resemble ours
  (the ball lost for long stretches, tracks dropping out, the ball pulled off the possessor) and
  scores a config with the official metric code, with a bootstrap interval.
  - The test the full match sets (prefer window 50 to 125) is only partly met. On one degradation
    seed (0), 50 scores 0.318 [0.292, 0.347], 125 scores 0.297 [0.264, 0.330] and 25 scores 0.272
    [0.252, 0.292]. So 25 is clearly worse than 50, but 50 versus 125 is not resolved: the
    intervals overlap and the gap is inside the noise band. The bootstrap resamples possessions,
    not degradation draws; the spread over several seeds is not measured yet.
  - The D truth is "no ball in the clean tracking", and `ball_visible` reads that same signal, so
    part of its alive/dead F1 (about 0.9) is built in. Methods that find stoppages inside ball gaps
    (e.g. from player speed) are tested only through the degradation assumptions, and player speed
    inside the degraded gaps is higher than on our data (2.4 against 1.7 m/s).
  - Limits: one match; degradation is seeded and parameterised, not measured (we cannot know
    which of our ball gaps are stoppages); corrected tracking pins the ball to the possessor, so
    the team stage is easier there than on automatic tracking. Use it to rank versions, not to
    forecast the score. Accept a change only beyond the noise band (spellF1 +/- 0.04).

## Reproducibility

`possession submit` runs the pipeline, re-parses the data from scratch, runs again, requires an
identical sha256, and calls the official validator. The CSV is written with explicit `\n` so the
bytes do not depend on the OS. `uv.lock` pins every dependency; `.python-version` pins 3.11.
