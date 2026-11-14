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
   The same file comes out of a fresh clone, a Linux arm64 container and a Linux x86-64
   container under emulation, each from an empty cache; Windows is not checked yet.
   The trained model is also committed (`models/`, added on 2026-11-14): with an empty cache
   the pipeline loads it, downloads nothing and writes the same file in 11 s; with
   `MTAH_NO_CACHE=1` it retrains from scratch and writes the same file in about two minutes.
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


## 8. Choosing between version 1 and version 3 (rule written on 2026-11-14, before measuring)

Two candidates are built for the last days. **Version 1** is the cautious one: the hybrid of
section 7, changed only by something that gains on the Metrica matches and loses nowhere.
**Version 3** is a bet on how the private ground truth reads "possession": it may lose under one
reading if it gains more under the other. This section fixes how the choice is made, so that the
choice cannot be fitted to the result. It replaces an earlier draft whose thresholds had been
written after seeing version 3's scores; an independent review of the plan pointed that out.

**What each set of matches is for.**

| Set | Role | Why |
|---|---|---|
| Metrica sample games, profiles `label_free` and `match_like`, readings `touch` and `controlled` | evidence | same provider as the competition tracking; the model has seen none of them |
| IDSSE, each match scored by a model trained without it, readings `controlled1.5` and `controlled3.5` | veto | another provider, ball in view during stoppages; but both readings lack very short possessions, which favours models trained on merged labels, so it cannot be evidence for version 3 |
| PFF, 16 held-out matches | veto | real tracking noise instead of a simulated one; the provider's own possession definition |
| The competition match, no labels | sanity ranges | the only look at the real tracker |
| `published_baselines` profile | reported only | it is fitted to scores computed from the private ground truth (RULES.md: nothing derived from it may be used), so it decides nothing from now on |

`match_like` is a degradation profile fitted to label-free statistics of the competition match
(how often our two tracks agree on possessions and on the team, the model's confidence about the
team, the share of frames without a ball). It takes the place `published_baselines` had.

**The rule.**

1. Weights of the two label readings: 50/50. The definition in CHALLENGE.md and the possession
   density of the public window and of the PFF labels (4.2 and 4.0 per live minute) speak for
   `controlled`; the published baseline scores are reproduced slightly better under `touch`.
   Hard rule 2 forbids choosing a version by the public window, so the default stays 50/50. Other
   weights are the owner's decision and have to be given before measuring.
2. The version 3 candidate comes from a family fixed here: training labels `raw`, `tau1`, `tau2`,
   decoder cost of a change of team 5, 8, 12, cost of a stoppage 12, 20; plus any PFF-trained
   variant that turns out to gain under one reading and lose under the other. The member is
   chosen on two Metrica matches and scored on the third. Nothing is tuned on IDSSE or PFF.
3. Version 3 is chosen if its weighted mean difference to version 1 over the evidence is at
   least 0.02 and it loses more than 0.02 in no evidence scenario (profile x reading).
4. IDSSE and the PFF test matches can each veto it: the mean difference there has to be at least
   0, with a win on most matches (5 of 7, 9 of 16).
5. Its label-free footprint on the competition match has to stay inside these ranges: 3.5 to 7
   possessions per live minute, dead share within 3 percentage points of version 1. In every
   other case version 1 is submitted.

**A change to version 1** needs: a gain of at least 0.02 on the Metrica matches under both
evidence profiles, no loss under either reading, a win on two of the three matches, and no loss
on IDSSE. A gain on the PFF test matches does not count for a model trained on PFF.

**Known before measuring.** The favourite of the version 3 family so far, labels `tau2` with
costs 8 and 20, loses 0.030 under `label_free` / `touch` (docs/DECISIONS.md, 2026-11-14), so
as things stand it fails point 3. Its training labels also contradict the
organisers' reference: they merge other-team spells under 2 s into their neighbours, and the
public window keeps such spells of 0.8 s and 1.3 s. The expected outcome is version 1.

## 9. Measured under the rule of section 8 (task 10, 2026-11-14)

**The profile `match_like`.** `possession calibrate --target match --write` searched 32 grid
points for the degradation under which `hybrid_supervised` and `unsup_v2` agree with each other
on the Metrica matches as they do on the competition match.

| statistic, no labels | competition match | `match_like` | `label_free` | PFF test matches |
|---|---|---|---|---|
| possessions of one pipeline scored against the other | 0.456 | 0.478 | 0.63 | 0.685 |
| same team where both say play | 0.829 | 0.814 | | 0.935 |
| mean team margin of the model | 0.678 | 0.678 | | 0.823 |
| share of frames without a ball | 0.407 | 0.483 | | 0.451 |

Fitted values: the ball is lost on 25% of live frames, its offset is 2.0 m, it stays in view
after 90% of stoppages for 1 to 6 s. `published_baselines`, fitted to something else entirely,
has the same values except an offset of 1.5 m. Under `match_like` the organisers' baseline
scores 0.213, `unsup_v2` 0.394 and the hybrid 0.524 (0.518 / 0.531; team accuracy 0.877).

**PFF as training data** (`scripts/variant_grid.py training label_free match_like` and
`... training pff`). spellF1 minus the submitted hybrid on the Metrica matches (touch /
controlled), and spellF1 on the 16 PFF test matches, which the 48 PFF training matches never
include.

| model trained on | cost of a change of team | `label_free` | `match_like` | mean of the four | 16 PFF test matches |
|---|---|---|---|---|---|
| IDSSE (submitted) | 3 | +0.002 / -0.030 | -0.022 / -0.053 | -0.026 | 0.772 |
| IDSSE (submitted) | 5 | 0 (0.659) | 0 (0.524) | 0 | **0.773** |
| IDSSE (submitted) | 8 | -0.021 / +0.027 | 0.000 / +0.034 | +0.010 | 0.755 |
| IDSSE and PFF | 3 | +0.003 / -0.011 | -0.039 / -0.061 | -0.027 | 0.880 |
| IDSSE and PFF | 5 | -0.006 / +0.017 | -0.024 / -0.019 | -0.008 | 0.874 |
| IDSSE and PFF | 8 | -0.024 / +0.031 | -0.015 / +0.017 | +0.002 | 0.855 |
| PFF | 3 | -0.022 / -0.027 | -0.098 / -0.114 | -0.065 | 0.911 |
| PFF | 5 | -0.029 / -0.003 | -0.081 / -0.083 | -0.049 | 0.905 |
| PFF | 8 | -0.042 / +0.013 | -0.067 / -0.043 | -0.035 | 0.890 |

A model that has seen PFF scores 0.10 to 0.14 higher on PFF at the same decoder cost, about the
same as the submitted model under `label_free` (+0.005 for IDSSE and PFF at cost 5) and lower
under `match_like` (-0.021 for IDSSE and PFF, -0.082 for PFF alone): it learns its provider, and
that costs where the tracking resembles ours. No variant reaches +0.02 on the Metrica matches, so
none changes version 1. One of them is a bet on the label reading in the sense of section 8
(IDSSE and PFF at cost 8: it gains under `controlled`, loses under `touch`), but its mean of
+0.002 is below every comparable member of the family below, so the choice of the version 3
candidate is the same with or without it. On the PFF matches the submitted model predicts 0.98
possessions per true one at a cost of 5 (run `20261114T160459*`).

**The version 3 family** (`scripts/variant_grid.py family label_free match_like`). spellF1 minus
version 1, by scenario; all 18 members are in docs/DECISIONS.md. The members that matter:

| labels, cost of a change, cost of a stoppage | `label_free` touch | `label_free` controlled | `match_like` touch | `match_like` controlled | mean | worst |
|---|---|---|---|---|---|---|
| raw, 8, 20 | -0.021 | +0.030 | +0.004 | +0.041 | +0.013 | -0.021 |
| tau1, 8, 20 | -0.018 | +0.034 | +0.009 | +0.051 | +0.019 | -0.018 |
| tau2, 5, 20 | -0.011 | +0.022 | +0.019 | +0.043 | +0.018 | -0.011 |
| **tau2, 8, 20** | -0.030 | +0.042 | +0.011 | +0.072 | **+0.024** | -0.030 |
| tau2, 12, 20 | -0.061 | +0.040 | -0.005 | +0.088 | +0.016 | -0.061 |

Chosen on two matches and scored on the third, the pick is `tau2, 8, 20` twice and
`tau1, 8, 20` once; held out it gains +0.019 on average (-0.028, +0.034, +0.008, +0.061). No
member has both a mean of +0.02 and a worst scenario above -0.02.

**The decision** (`possession decide configs/hybrid_supervised.toml configs/hybrid_bet.toml`, run
`experiments/runs/20261114T160229*`; `hybrid_bet` is `tau2, 8, 20`).

| condition for version 3 | measured | passes |
|---|---|---|
| weighted mean over the evidence at least +0.02 | +0.024, 5-95% interval over (match, window) blocks +0.014 to +0.035 | yes |
| no evidence scenario below -0.02 | -0.030 under `label_free` / `touch` | **no** |
| veto IDSSE: mean at least 0, 5 of 7 matches | +0.020 (-0.018 / +0.057), 7 of 7 | yes |
| veto PFF: mean at least 0, 9 of 16 matches | -0.052, 0 of 16 | **no** |
| footprint: 3.5 to 7 possessions per live minute | 4.10 (version 1: 5.59) | yes |
| footprint: dead share within 3 points of version 1 | 0.4864 against 0.4935 | yes |

**Version 1 is submitted.** Version 3 would make 223 possessions on the match instead of 300
and differs from version 1 on 3.6% of frames.

**The change to version 1 that `published_baselines` alone had rejected** (section 6: our
label-free evidence added at half weight in both stages, `configs/hybrid_plus_unsup.toml`):

| set | difference to the hybrid | matches won |
|---|---|---|
| Metrica, `label_free` | +0.022 (touch +0.022, controlled +0.022) | 3 of 3 |
| Metrica, `match_like` | -0.021 (-0.018, -0.025) | 0 of 3 |
| IDSSE, out of fold | -0.012 (-0.013, -0.011) | 0 of 7 |
| PFF test matches | +0.009 | 9 of 16 |

It helps where the tracking is easy and hurts where it resembles ours, so it stays out, now
without any use of the published scores.

**A correction to section 8.** The rule quotes 4.0 possessions per live minute for the PFF
labels. That was one match; over all 64 it is 4.7 (mean possession 13.1 s). The rule itself does
not depend on the figure.

## 10. Stoppages: training regime, test-time augmentation, a temporal model

Written on 2026-11-14. The family and the rule below were fixed before any member was measured.
The gates were evaluated in task 11, and G2's statistic was replaced during it, after the match's
moving-ball share had been seen ("The gates", below; docs/DECISIONS.md, task 11).

**Why.** On the 16 PFF test matches, another team's ball-state evidence in place of ours (our team
evidence and decoder kept) raises spellF1 from 0.773 to 0.80-0.84, and true stoppages with our team
evidence would give +0.07 on both Metrica profiles. Our training matches (IDSSE as degraded by the
package) lose the ball without regard to stoppages; automatic tracking loses it during them
(docs/DECISIONS.md, task 11). Gate G2 of task 11 found that the competition match loses the ball
almost only during stoppages but keeps it in view during about a quarter of its stoppage time, so
of the regime variants only the mixed one is measured.

**The family** (fixed; the team evidence is version 1's in every member except M4):

| member | configuration | what changes |
|---|---|---|
| M3 | `configs/hybrid_regime_mix.toml` | ball-state evidence from a LightGBM trained with noise seed 0 as before and seed 1 in the automatic-ball regime `auto_lf` (`possession/regime.py`) |
| M4 | `configs/hybrid_tta.toml` | version 1 with test-time augmentation in both stages: the prediction averaged with that of the match with home and away exchanged |
| M5 | `configs/hybrid_sequence_blend.toml` | ball-state evidence: version 1's LightGBM and a temporal network (task 13, trained in the M3 regime) at weights 0.5 and 0.5, plus the kick-off rule |
| M6 | `configs/hybrid_sequence.toml` | ball-state evidence: the temporal network alone, plus the kick-off rule |

M1 and M2 (regimes `auto_lf` and `auto_ml` alone) are dropped by gate G2 and are not measured. M5
and M6 exist only if task 13 passes its numerical check (G4).

**The rule** (a change to version 1, section 8, made precise):

1. Evidence: the three Metrica matches under `label_free` and `match_like`, readings `touch` and
   `controlled`. The member is chosen on two matches (mean over both profiles and both readings) and
   scored on the third; the decision uses these held-out differences, not the in-sample ones.
2. The held-out gain must be at least +0.02 under each of the two profiles, with no loss under
   either reading in either profile, and the member must win on at least two of the three matches.
3. Vetoes: IDSSE out of fold, plain inputs (suite `idsse`): mean difference at least 0 and 5 of 7
   matches won; PFF test matches: mean difference at least 0 and 9 of 16 won. The suite
   `idsse_auto` (the held-out matches in the automatic-ball regime of `match_like`) is reported only.
4. Footprint on the competition match: 3.5 to 7 possessions per live minute and a dead share between
   0.35 and 0.50 (the true dead shares of the three labelled sets). The earlier "within 3 points of
   version 1" would block exactly this kind of change.
5. Otherwise version 1 stays. Every member is logged in docs/DECISIONS.md whatever its result.

**Mechanism check, not a decision (G3).** M3 should cut the stoppages it invents with the ball
tracked on PFF; if it does not, only bugs are fixed, nothing is tuned.

**The gates**, as far as the repository records them:

- **G1 (oracle bound).** spellF1 with the true stoppages as ball-state evidence (the hybrid's own
  team evidence and decoder, `scripts/deadball_errors.py`) minus the hybrid's, per set. No
  numerical threshold was written down before measuring. Against the noise band (±0.04): Metrica
  `label_free` +0.070, `match_like` +0.072 and PFF +0.100 clear it; IDSSE out of fold +0.038 does
  not.
- **G2 (regime of the match).** Fixed during task 11, after the planned moving-ball statistic
  failed to separate true from invented stoppages. Statistic S: the share of frames with running
  players (90th percentile of their speeds, 1 s mean, at least 2.5 m/s). f = (S_match - S_dead) /
  (S_live - S_dead), the share of play, with S_dead and S_live from the labelled sets with tracked
  players (Metrica, IDSSE). f < 0.5: only M3 is run and the IDSSE veto uses plain `idsse`. The
  other branch (M1 and M2 run) is not written down further. The references quoted so far are for
  all true stoppages, not for stoppages with a tracked ball (docs/DECISIONS.md, task 11).
- **G3.** The mechanism check above.
- **G4.** Task 13's numerical check, defined in `tasks/13-sequence-model.md`: the numpy inference
  reproduces torch's float64 output on a stored input to 1e-9.
