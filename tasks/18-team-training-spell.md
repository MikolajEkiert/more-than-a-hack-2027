---
status: doing
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
- [ ] new options in the team training path, off by default; version 4's cache key and committed
      model unchanged (checked), tests for the new weight function and TTA wiring
- [ ] 8 members: final models + IDSSE out-of-fold models trained (2 threads), all 7 sets measured
- [ ] rule applied (truth:c14_best, unit, vs version 5); verdict written here and in DECISIONS
- [ ] runs logged (`possession validate --note`) for the chosen member (or version 4 if it stays)
- [ ] `make check` green; committed on `task/18-team-training-spell`

## Log
- 2026-11-15 · Claude · task opened; family and rule written before measuring.
