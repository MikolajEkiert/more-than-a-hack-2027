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

## Several sources for one stage

`ball_state.blend` and `team.blend` combine sources (registered methods, or files of per-frame
probabilities from a model outside the registry) as a weighted sum of log-odds. This is how a
supervised model is fused with the label-free evidence: before the decoder, so the output is
still whole possessions. Contract and decision protocol: docs/FUSION.md.

## The supervised model

`solution/` is the supervised track's package (branch `supervised_approach`, merged as written):
seven open Bundesliga matches (IDSSE) degraded to look like automatic tracking, 203 hand-made
features, one LightGBM with the classes H / A / D. `possession/supervised.py` is the only module
of the pipeline that imports it. It converts a `Match` into the package's layout, trains the
package's final model once, and returns three class probabilities per frame. The methods
`ball_state.supervised` (1 - P(D)) and `team.supervised` (P(H) / (P(H) + P(A))) read them; their
parameter `labels` names the label function the model is trained on.

- The pipeline takes the package's features and model, nothing else. Its decoder, its command
  line and its kick-off rule are not called.
- The degradation of the training matches is calibrated on three unlabelled statistics of the
  competition tracking, so the model needs the match data even when it scores another match.
- The open data and the package's cache live in `.cache/idsse/` and `.cache/solution/`, never in
  `data/`. Both can be deleted and are rebuilt identically: 2.63 GB of downloads (one fixed
  revision, each file checked against `solution/idsse.sha256`), then about three minutes on a
  laptop.
- `solution/` and `notebooks/` are excluded from ruff, like organiser code. The package's cache
  keys hash its own source, and a diff there should mean a real change.
- Unit tests replace the model with a stub (`tests/conftest.py`), so the methods and the hybrid
  config are covered on a fresh clone. The model itself is checked by retraining from an empty
  cache and comparing the file (docs/FUSION.md section 7).

## Evaluation without tuning on the public window

- `scoring.diagnostics`: possession count, mean duration, home share and label shares, all with no GT.
- `scoring.public_sanity`: the public 5 minutes; a bug detector only.
- `validation` (`possession validate CONFIG... [--against CONFIG] [--profile NAME]`): the
  evidence for choosing between versions, within the limits below.
  - **Matches**: Metrica's public Sample Games 1-3 (same provider as the competition tracking;
    game 3 in the same EPTS format, games 1-2 in CSV). Downloaded on first use, never committed.
  - **Labels**, derived by an algorithm: D where the corrected tracking has no ball and the events
    confirm a stoppage; otherwise the team in possession under two readings, both always scored:
    `touch` (the team of the last on-ball event: the provider's event definition) and `controlled`
    (a touch counts only if the team then holds the ball 0.2 s or completes a pass, shot or set
    piece: CHALLENGE.md's definition). `controlled` gives 4.03-4.09 possessions per live minute
    and a mean of 14.7-14.9 s on the three matches, against 4.2 and 14.2 s published for the
    competition's public window; `touch` gives about 6.0 and 10.0-10.5 s. The published
    full-match baseline scores are reproduced slightly better by `touch`. Which one the ground
    truth follows is unknown, so a change must not lose under either.
  - **Degradation**: a named profile in `configs/degradation/` (see its README). `label_free` is
    set from label-free aggregates of our tracking; `published_baselines` is fitted so that the
    organiser baselines score what EVALUATION.md publishes for the full match, and is harsher.
    One degraded match serves both label readings; 3 seeds per match.
  - **Acceptance rule, fixed in advance** (`validation.compare`, printed by `--against`): mean
    spellF1 gain of at least 0.02, no loss under either label reading, and a win on at least two
    of the three matches.
  - Where it stands: `baseline_smooth50` scores 0.334 under `label_free` and 0.238 under
    `published_baselines` (0.2433 on the real match). Window 50 beats 125 on the overall mean
    under `label_free` and under `touch`, as on the full match; under `controlled` 125 is ahead.
  - Limits: three matches of one provider; the degradation is seeded and parameterised, not
    measured (we cannot know which of our ball gaps are stoppages); the D truth is "no ball in the
    clean tracking", which `ball_visible` partly reads back; corrected tracking pins the ball to
    the possessor, and how far the degradation undoes that is an assumption. Use it to rank
    versions, not to forecast the score.

## Reproducibility

`possession submit` runs the pipeline, re-parses the data from scratch, runs again, requires an
identical sha256, and calls the official validator. The CSV is written with explicit `\n` so the
bytes do not depend on the OS. `uv.lock` pins every dependency; `.python-version` pins 3.11.
