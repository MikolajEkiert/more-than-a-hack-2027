# Combining the label-free track with a supervised model

Two tracks are being built: the label-free pipeline (`configs/unsup_v2.toml`) and a supervised
model trained on other matches. This document is the contract between them: what the supervised
track has to deliver, how the two are combined, and how the team decides between fusion and
choosing one output. Nothing here depends on how the supervised model is built.

## 1. Why fuse evidence, not labels

Both tracks end in the same decoder. The decoder takes, per frame, evidence that play is live
(`p_alive`) and evidence for the home team (`p_home`), and returns whole possessions.

- **Fusion happens before the decoder**: each stage's evidence is a weighted sum of the log-odds
  of its sources (`ball_state.blend`, `team.blend`). The output is still one coherent sequence.
- **Voting on two label files frame by frame is not an option.** Wherever the tracks disagree for
  a few frames it creates short possessions, and spellF1 counts every one of them.
- If only a finished label file is available from the supervised track, the choice is between
  the two files (section 4), not a fusion.

## 2. What the supervised track delivers

Either of these, per frame of a match:

| quantity | meaning | range |
|---|---|---|
| `p_alive` | probability that the ball is in play | [0, 1], no NaN |
| `p_home` | probability that the home team has the ball, given that it is in play | [0, 1]; NaN = no opinion |

or three class probabilities `p_h`, `p_a`, `p_d`, from which `p_alive = 1 - p_d` and
`p_home = p_h / (p_h + p_a)` are derived.

Two ways to plug them in:

1. **Registered methods (preferred).** `@register("ball_state", "<name>")` and
   `@register("team", "<name>")` in `src/possession/stages/`, taking a `Match` and returning the
   arrays above (`src/possession/AGENTS.md` has the contract). They then run on any match,
   including every degraded external match, and the repository regenerates the submission by
   itself. Model weights are either trained by a deterministic script in the repo or committed.
2. **Files.** One `.npz` (or `.npy`) per match, read by a source `{ file = "path/{game}.npz" }`.
   `{game}` is the match's name: `BAR-ATH` for the competition match, and
   `<GAME>@<profile>#<seed>` for a degraded external match (for example
   `METRICA-SG3@label_free#0`). `uv run possession export` writes the tracking of every validation
   condition under exactly those names (to `.cache/exports/`, no labels), so a model kept outside
   the repo can be run on the same inputs. Files are not committed; for the final submission the
   repo must still be able to regenerate them (RULES.md 3).

Requirements either way:

- **Leave-one-match-out.** Validation uses Metrica Sample Games 1-3. A score on one of them only
  counts if the model was trained without that match, in any degraded form. If the model trains on
  the Metrica games at all, it needs a way to train with one held out.
- **Label reading.** External labels exist in two readings, `touch` and `controlled`
  (`validation.CONVENTIONS`, docs/ARCHITECTURE.md). Say which one the model was trained on.
  `validation.load_external(game, convention)` returns the labels in either, and
  `validation.event_owner` is the derivation, so both tracks can use the same one.
- **Inputs it will see.** Automatic tracking: the ball missing for whole stretches of 5 s or more
  (41% of graded frames), about 20.5 players visible, no identities. `validation.degrade` makes an
  external match look like that; `possession inspect ball|players|gaps` shows the aggregates.
- **No fitting on the competition match** unless the team has cleared it against the organisers'
  email of 3 November (docs/research/UNSUPERVISED_RESEARCH.md section 6).

## 3. A fusion config

```toml
name = "fusion"

[ball_state]
method = "blend"
sources = ["stoppage_evidence", "supervised"]          # or { file = "probs/{game}.npz" }
weights = [1.0, 1.0]

[team]
method = "blend"
sources = ["touch_episodes", "supervised"]
weights = [1.0, 1.0]

[decode]
method = "switch_cost"
turnover_cost = 5.0
stop_cost = 12.0
restart_cost = 12.0
snap_start_s = 3.0
snap_end_s = 3.0
```

A source is a method name, a table `{ method = "...", <its parameters> }`, or a file table. A
weight of 2 counts a source twice; 0 silences it. For the team stage a source may abstain (NaN);
the result is NaN only where every source abstains, and the decoder then keeps the last owner.
`configs/blend_team_example.toml` is a working example with two of our own sources.

## 4. How to decide (fixed before measuring)

Everything is scored on the external matches, under both label readings and under both
degradation profiles (`possession validate ... --profile label_free|published_baselines`). The
public 5-minute window is a sanity check only and decides nothing.

1. Score the two tracks alone:
   `make validate CONFIGS="configs/supervised.toml" AGAINST=configs/unsup_v2.toml`.
2. Try fusion with a small grid of weights (0.5, 1, 2 for the supervised source, per stage) and
   re-tune the three switching costs once.
3. **Fusion is used only if** it passes the acceptance rule (`validation.compare`: mean gain of at
   least 0.02, no loss under either label reading, a win on at least two of three matches)
   against the better single track, under both profiles.
4. **Otherwise the better single track is used**, by the same rule. If neither beats the other by
   the rule, the label-free track is used: it has no training data to document and no domain gap.
5. Stages may be decided separately: for instance the supervised `p_home` with our `p_alive`.

What to expect, from measurements on the label-free track with one stage replaced by the truth
(game 3, `controlled` reading): a perfect team signal on frames with a tracked ball lifts 0.65
to 0.84; perfect stoppages to 0.73; a perfect team signal on frames without a ball to 0.73. The
team in control of a visible ball is where a supervised model can add most.

Two sources that read the same signal add nothing: blending our two team methods scores 0.589
against 0.598 for `touch_episodes` alone and is rejected by the rule. Fusion pays when the second
source has seen something the first has not.

## 5. Checklist for the report

For whichever combination is submitted, `docs/TECHNICAL_REPORT.md` section 3 lists every dataset
the supervised model was trained on (source, who labelled it, how much), and section 5 cites the
`experiments/runs/` files of the comparison above.

## 6. Measured on 2026-11-14 against the branch `supervised_approach`

What was run. The branch at commit 8db66f2 (package `solution/`), trained exactly as its notebook
does: seven IDSSE matches, degradation calibrated on the unlabelled tracking of the match (ball
missing 40.7%, mean burst 17.2 s, 93% of players visible), 203 features, weighted LightGBM for
H/A/D plus its boundary model. `scripts/supervised_probs.py` wrote its probabilities for the
competition match and for all 18 validation conditions. It trains on no Metrica match, so all
three are held out for it. Retrained from scratch with its cache disabled, it gave bit-identical
probabilities.

Mean spellF1 over three external matches x 3 seeds x 2 label readings (touch / controlled):

| variant | `label_free` | `published_baselines` |
|---|---|---|
| `unsup_v2` (ours) | 0.598 (0.560 / 0.636) | 0.448 (0.418 / 0.477) |
| supervised as is, its own decoder | 0.621 (0.634 / 0.608) | 0.506 (0.517 / 0.494) |
| **`hybrid_supervised`**: its evidence, our kick-off rule, our decoder | **0.659** (0.643 / 0.675) | **0.561** (0.546 / 0.576) |
| hybrid + our evidence at half weight, both stages | 0.681 (0.665 / 0.697) | 0.553 (0.545 / 0.561) |
| our `p_alive` + its `p_home` | 0.656 | 0.549 |
| its `p_alive` + our `p_home` | 0.600 | 0.448 |

Verdicts of the acceptance rule (`validation.compare`):

- `hybrid_supervised` against the supervised branch as is: +0.038 (touch +0.009, controlled
  +0.066, 3 of 3 matches) and +0.055 (+0.029, +0.081, 3 of 3): **accepted** under both profiles.
- `hybrid_supervised` against `unsup_v2`: +0.061 and +0.113: **accepted** under both.
- Adding our evidence at half weight to the hybrid: +0.022 under `label_free`, -0.007 under
  `published_baselines`: **not accepted** (it must pass under both).
- The branch's boundary model as a frame-varying turnover cost in our decoder: between -0.006
  and +0.010: **not accepted**.

What it says:

- **The supervised evidence is better than ours, and the difference is the team stage.** Team
  accuracy 0.920 against 0.894 (`label_free`) and 0.888 against 0.826 (`published_baselines`);
  alive/dead F1 is the same in both tracks (0.934 / 0.932 and 0.927 / 0.916). Swapping stages
  shows it directly: its `p_home` with our `p_alive` scores like the hybrid, its `p_alive` with
  our `p_home` scores like `unsup_v2`.
- **Our decoder adds to its evidence.** With the settings of `unsup_v2`, not re-tuned, its
  evidence scores 0.04-0.05 above its own decoder. Its decoder was tuned on IDSSE, whose owner
  flag changes hands twice as often as the Metrica labels; it predicts 328-392 possessions
  where ours predicts 275-299 on the same evidence.
- **A worry that did not hold:** its degradation hides the ball at random, so it might not have
  learned that a long ball gap is usually a stoppage. It has: on the frames of our match
  without a ball its mean P(alive) is 0.14 (ours 0.15).
- **On the competition match** (label-free): supervised as is 352 possessions, mean 9.3 s, 48%
  dead; hybrid 300, 10.7 s, 49%; `unsup_v2` 326, 10.8 s, 44%. The two tracks give the same label
  on 83% of graded frames. Without a kick-off rule the supervised output is live for 117 s before
  the second-half kick-off; with `ball_state.kickoff_wait` as a source the hybrid starts 0.2 s
  from the published anchor, with no constant of this match.

By the protocol of section 4 the hybrid is the team's candidate. When this section was first
written it could not be submitted, because `configs/hybrid_supervised.toml` read probability
files that only existed locally. Section 7 records how that was closed.

## 7. The hybrid, reproducible from the repository (task 08, 2026-11-14)

1. **The model is trained by the repository.** The supervised branch is merged as the package
   `solution/`, as written. The pipeline reaches its model through `possession/supervised.py`
   and two registered methods, `ball_state.supervised` and `team.supervised`; LightGBM 4.5.0 and
   SciPy 1.17.1 are pinned in `pyproject.toml` and `uv.lock`. The first run downloads the open
   data (2.63 GB, one fixed revision, every file checked against `solution/idsse.sha256`) and
   trains; later runs load the cached model.
2. **Nothing changed in the output.** Retrained from an empty cache under the repository's lock
   (numpy 2.4.6; the branch's own environment had 1.26.4), the model file is byte-identical to
   the one the branch trained, the submission file has the same sha256 as before the port, and
   the external scores are the same: 0.659 (`label_free`) and 0.561 (`published_baselines`),
   accepted against `unsup_v2` under both. Runs: `experiments/runs/20261114T012451*` (the
   match), `T012556*` and `T012701*` (external).
3. **Only the model is taken from the package.** Its own decoder, command line and kick-off rule
   (which reads the organisers' published kick-off frame) stay in `solution/` for the supervised
   track's notebooks, but the pipeline calls none of them. The kick-off rule in use is
   `ball_state.kickoff_wait`, from the formation, with no constant of this match (AGENTS.md hard
   rule 1).
4. **The training data** is listed in `docs/TECHNICAL_REPORT.md` section 3.
5. **Fitting on the match.** The model's degradation is calibrated on three unlabelled
   statistics of the competition tracking (share of frames without a ball, mean length of those
   stretches, players visible). No label of the match is involved, and RULES.md allows it ("the
   provided data is a starting point"). The team received no message from the organisers that
   says otherwise (docs/DECISIONS.md).
6. **The turnover cost stays at 5**, the value that does not lose under either label reading (5
   favours `touch`, 8 favours `controlled`). The label function the model is trained on is a
   parameter (`labels`), `raw` by default: the two alternatives measured so far gain under one
   reading and lose under the other (docs/DECISIONS.md).

