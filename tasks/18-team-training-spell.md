---
status: done
owner: MikolajEkiert (Claude)
stage: team
---

# 18 · Train the team model for spellF1: weights at possession changes, debounced training labels, home/away test-time averaging

## Goal
The team decision on frames with a ball is the largest remaining loss (oracle analysis, task 14/15).
Version 4's team evidence is the LightGBM "v1+team" trained per frame with the package's sample
weights. Train it so that what spellF1 counts weighs more (the frames around a hand-over), so that
its targets are possessions rather than touches (debounced labels), and average its prediction over
the home/away-swapped view at inference. Each is measured as a family member against version 4 on
every validation set; version 4 stays unless the rule below says otherwise.

## Context
- `supervised._train_final_team` / `_team_folds` train "v1+team" exactly as the package trains its
  model: 7 IDSSE matches, noise seeds 0 and 1, both orientations, every 5th graded frame, the
  package's sample weights (`ev.sample_weights`, `DEFAULT_CFG`: +5 within 10 frames of any label
  change, D edges included, +3 inside spells shorter than 5 s), LightGBM 300 rounds.
- Ideas of other teams, credited (research/COMPETITORS.md section 4): boundary-weighted loss (Team
  Sigue), short-possession weights (SAM; Mantas weights hand-over frames x40), a "controlled"
  relabelling of the training labels with a 2-3 s debounce (many teams), home/away swap as
  test-time averaging (CoreOrigin, Sigue, O(1)Burger, SAM). The package's weights already are a
  first version of the first two; the new knob here is a multiplier on a wider window around
  team changes only (H to A or A to H).
- Tried and dropped (docs/TECHNICAL_REPORT.md section 2): "training labels with short
  interruptions merged and a slower decoder" (the package's model, `tau2`, decoder costs 8/20). Not
  retried as such: here the debounce is the package's `controlled<s>` relabelling, on the "v1+team"
  model, with version 4's decoder unchanged. Test-time averaging was measured on version 1 (M4,
  FUSION.md section 10), never on "v1+team".
- Caveat fixed before measuring: the IDSSE validation readings `controlled1.5` / `controlled3.5` are
  the package's `controlled_possession` applied to the provider's flags, the same function the
  debounced members train on. A member trained on `controlled1.5` is therefore favoured on the IDSSE
  `controlled1.5` reading (still out of fold: other matches). IDSSE only acts as a floor in the rule,
  never as evidence.

## Scope
- May touch: `src/possession/supervised.py` (team training path: new options, off by default),
  `scripts/team_block.py` (training driver options), `configs/t18_*.toml`, tests, this file,
  docs/DECISIONS.md, docs/FUSION.md, experiments/runs/.
- Out of scope: `solution/`, `evaluation/`, `starter/`, `regime.py`, `sequence.py`,
  `sequence_train.py`, the decoder, the ball-state stage. Version 4 (`configs/c14_best.toml`) must
  stay byte-identical and keep loading `models/team/57471b9e1a31bec96010.txt.gz`.

## Interface
- `FEATURE_SETS` gains members with a training option `turn_weight` (multiplier m) and `turn_s`
  (half-window in seconds): every training row within `turn_s` of a team change of the training
  labels has its package weight multiplied by m. New cache keys; "v1+team" unchanged.
- Debounce: the existing `labels` parameter, `labels = "controlled1.5"` / `"controlled2.5"`
  (`solution/labels.controlled`: a hand-over counts only if the new team keeps the ball that long;
  spells right after a stoppage always stay).
- `tta = true` for "v1+team" sets: P = mean of the original view and the home/away-swapped view
  (columns swapped back), final model and out-of-fold models alike.

## Family (fixed before measuring; K = 8)
| member | labels | turn weight | TTA |
|---|---|---|---|
| `t18_tw2` | raw | x2 within +-1 s | no |
| `t18_tw4` | raw | x4 within +-1 s | no |
| `t18_db15` | controlled1.5 | - | no |
| `t18_db25` | controlled2.5 | - | no |
| `t18_tta` | raw (version 4's model) | - | yes |
| `t18_tw2_tta` | raw | x2 within +-1 s | yes |
| `t18_db15_tta` | controlled1.5 | - | yes |
| `t18_db15_tw2` | controlled1.5 | x2 within +-1 s | no |

Everything else is version 4's (`c14_best`): ball state, decoder, LightGBM parameters, rounds.

## Decision rule, written before any number of this task was measured (2026-11-15)
Rule D of task 15 with its final weighting, reference version 4:
`uv run python scripts/grid14.py decide-d <8 members> --reference c14_best --bins truth:c14_best`.
1. Evidence: Metrica `label_free`, `match_like`, `match_regime`, `match_regime_still`; readings 2/3
   `controlled`, 1/3 `touch`; windows weighted by the match's stoppage-density bins (windows binned
   by their truth, the match by version 4's labels). A member is chosen on two Metrica matches and
   scored on the third.
2. Accept if the held-out weighted gain >= max(0.005, 0.003 x sqrt(2 ln K)) = +0.0061 (K = 8) and
   the 5% bound of the weighted block bootstrap is above 0.
3. Floors: no (profile, reading) below -0.010; PFF >= -0.005; plain IDSSE and `idsse_auto` >= -0.010
   (weighted, for the member chosen on all three matches).
4. Footprint on the match inside (3.5 to 7 possessions per live minute, dead share 0.35 to 0.50).
5. Robustness: the same member must also pass with every window weighing 1 (`--bins unit`).
   Otherwise version 4 stays.
6. Reported, not deciding: the same comparison against version 5 (`d15_s28_l6_t6`, version 4 with
   turnover cost 6). If a member passes against version 4 and its gain does not hold against version
   5, that is stated; a member + turnover 6 combination would be a new task, not adopted here.

## Acceptance criteria
- [x] new options in the team training path, off by default; version 4's cache key and committed
      model unchanged (checked), tests for the new weight function and TTA wiring
- [x] 8 members: final models + IDSSE out-of-fold models trained (2 threads), all 7 sets measured
- [x] rule applied (truth:c14_best, unit, vs version 5); verdict written here and in DECISIONS
- [x] runs logged (`possession validate --note`) for the chosen member (or version 4 if it stays)
- [x] `make check` green; committed on `task/18-team-training-spell`

## Log
- 2026-11-15 · Claude · task opened; family and rule written before measuring (commit ab8b184).
- 2026-11-15 · Claude · Implemented (supervised.py): `FEATURE_SETS` "v1+team-tw2" / "v1+team-tw4"
  (`turn_multiplier`, `_team_turn_rows`, `_team_rows_of`; the hashed training functions are
  untouched and the options enter the key only for the new sets), debounce through the existing
  `labels` parameter, `tta` for the team sets in `_predict_team` and `_team_folds` (the swapped
  view has its own fold file; the original view is the plain prediction's file). Version 4's team
  key is still 57471b9e1a31bec96010 and `possession run configs/c14_best.toml` still gives sha256
  6fd87cc4467530fa. Tests: turn multiplier, row alignment, TTA wiring.
- 2026-11-15 · Claude · Trained at OMP_NUM_THREADS=2 (`scripts/team_block.py train --features F
  --labels L [--tta]`): final model 142-236 s, 2 x 14 fold models 547-738 s per seed, 1190-1500 s
  per member; five trainings plus version 4's swapped-view folds (1191 s). Cache keys of the final
  models: tw2 5032aff70544690fd7fc, tw4 b9337415b884b4faffe7, db15 0719e530a6631ac807e6, db25
  d381b1346f3ca2f18c66, db15+tw2 38013a8b48f7c86bce5f. Caveat: the IDSSE fold models of the new
  members (and the swapped view of `t18_tta`) were trained at 2 threads, version 4's at 4/8; this
  adds a little noise to the IDSSE differences, none to Metrica and PFF.
- 2026-11-15 · Claude · Measured all 8 members on the 7 sets (`grid14.py measure`). Difference to
  version 4, unweighted (5-95% interval over (match, window) blocks), matches won:

  | member | label_free | match_like | match_regime | regime_still | PFF | IDSSE | idsse_auto |
  |---|---|---|---|---|---|---|---|
  | tw2 | -0.011 | -0.016 | -0.006 | -0.006 | -0.004 [-0.009, +0.001] 8/16 | -0.009 0/7 | -0.007 1/7 |
  | tw4 | -0.023 | -0.035 | -0.023 | -0.023 | -0.020 0/16 | -0.016 0/7 | -0.011 1/7 |
  | db15 | +0.001 | +0.033 | +0.009 | +0.011 | -0.027 [-0.032, -0.020] 2/16 | +0.017 7/7 | +0.017 7/7 |
  | db25 | -0.012 | +0.033 | -0.002 | +0.001 | -0.078 0/16 | +0.017 7/7 | +0.020 6/7 |
  | tta | -0.002 | +0.002 | +0.001 | +0.002 | +0.001 [-0.001, +0.003] 12/16 | -0.002 3/7 | +0.002 5/7 |
  | tw2_tta | -0.008 | -0.013 | -0.004 | -0.003 | -0.003 4/16 | -0.008 0/7 | -0.004 1/7 |
  | db15_tta | +0.000 | +0.031 | +0.009 | +0.012 | -0.026 2/16 | +0.016 6/7 | +0.020 7/7 |
  | db15_tw2 | +0.000 | +0.025 | +0.010 | +0.012 | -0.019 2/16 | +0.012 6/7 | +0.012 7/7 |

  By reading: the turn weight loses under `controlled` (tw2 -0.013 to -0.025, tw4 -0.039 to
  -0.046) and is about even under `touch`: weighting the frames around hand-overs makes the model
  change team more readily, i.e. more short possessions. The debounce gains under `controlled`
  (db15 +0.012 to +0.050, db25 +0.030 to +0.076) and loses under `touch` (db15 -0.002 to -0.010
  except match_like +0.016; db25 -0.046 to -0.053) and on PFF, whose labels change owner at every
  on-ball event. IDSSE's gain is mostly `controlled3.5` (db15 +0.030, db25 +0.074; db25 -0.040
  under `controlled1.5`), and the `controlled1.5` reading is the very function db15 trains on.
  TTA moves nothing beyond noise (|diff| <= 0.0024 everywhere; +0.003 on top of tw2).
- 2026-11-15 · Claude · Rule D (`decide-d`, K = 8, threshold +0.0061). With `--bins truth:c14_best`:
  held-out picks t18_db25, t18_db25, t18_db15; held-out weighted gain +0.0161, 5% bound +0.0069,
  but every `touch` cell is below the -0.010 reading floor except match_like (label_free -0.035,
  match_regime -0.041, match_regime_still -0.037, match_like -0.004); member chosen on all three
  matches t18_db25: PFF -0.087 (floor -0.005) FAIL, IDSSE +0.036, idsse_auto +0.039, footprint
  inside (4.10 possessions per live minute, dead 0.479). **Version 4 stays.** `--bins unit`: the
  same picks, gain +0.0151 (bound +0.0063), the same floors fail, PFF -0.075. Against version 5
  (`d15_s28_l6_t6`): gain +0.0061 / +0.0060 with the 5% bound below 0 (-0.0021 / -0.0017), PFF
  -0.080 / -0.071: version 5 would stay too. Outside the rule, the best-balanced member db15_tw2
  still loses 0.019 on PFF (0.025 weighted), 2 of 16 matches, so no member of this family is a
  candidate under any reading that counts PFF.
- 2026-11-15 · Claude · Conclusion: training the team model toward possessions (debounce) is the
  same bet on the `controlled` reading as the decoder's turnover cost 6 (version 5) and the dropped
  `tau2` labels, only larger, and PFF refuses it; the hand-over weights make the model more eager
  to switch and lose everywhere; test-time averaging is neutral on "v1+team" as on version 1 (M4).
  Nothing to retrain. Runs: experiments/runs/20261115T1535*_t18_* (match_like) and 20261115T1544*_t18_* (PFF), against version 4.
