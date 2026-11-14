---
status: review
owner: MikolajEkiert (Claude)
stage: infra, ball_state, team
---

# 10 · PFF as training and test data, a label-free harsh profile, and the choice between versions

## Goal
Answer three questions with measurements, under the rule fixed in docs/FUSION.md section 8:
does PFF help as training data; which candidate is version 3 and does it beat version 1; does
anything else that was rejected only by the `published_baselines` profile deserve to be in.

## Context
The PFF World Cup 2022 bundle (64 matches of broadcast tracking with an event stream, free but
behind the provider's request form) is in `.cache/pff/`. Its tracking loses the ball on about 44%
of frames, in gaps that are almost all longer than 3 s, like the competition tracking; its labels
change hands 4.7 times per live minute (public window: 4.2). Players are mostly estimated there
(broadcast), which is unlike our tactical tracking.

## Scope
- May touch: `src/possession/supervised.py` (PFF conversion, split, training variants, test
  conditions), `stages/ball_state.py` and `stages/team.py` (parameter `data`),
  `src/possession/validation.py` (suite `pff`, windows, choice rule, calibration to the match),
  `src/possession/scoring.py` (footprint), `src/possession/inspect_data.py` (kick-off audit),
  `src/possession/cli.py`, configs, tests, docs.
- Out of scope: changes to `solution/`; the final submission (task 06).

## Interface
- `ball_state.supervised(labels, data)`, `team.supervised(labels, data)`; `data` is `idsse`,
  `idsse+pff` or `pff`.
- `possession validate --suite pff`: the 16 PFF matches kept out of every training set, with
  their tracking as delivered.
- `possession calibrate --target match [--write]`: the profile `match_like`.
- `possession decide V1.toml V3.toml`: evidence, vetoes, footprint, verdict.
- `possession inspect kickoffs`: the kick-off rule against Smart Tagging.

## Acceptance criteria
- [x] the PFF split is fixed before measuring and a test match is never trained on (tested)
- [x] PFF as training data measured on the Metrica matches and on the PFF test matches
- [x] `match_like` fitted without labels or published scores
- [x] version 3 chosen from its family by nested selection and compared with version 1 by
      `possession decide`; the decision logged
- [x] the addition that `published_baselines` alone had rejected measured under the evidence
      profiles and on IDSSE
- [x] kick-off rule audited against Smart Tagging

## Log
- 2026-11-14: PFF converted (64 matches) and split 48 / 16 before any measurement. Bridge:
  `convert_pff`, `pff_games`, `model(labels, data)`, `pff_conditions`; suite `pff`.
- 2026-11-14: `possession calibrate --target match --write` gave `match_like` (ball lost on 25% of
  live frames, offset 2.0 m, ball in view after 90% of stoppages for 1-6 s). It lands next to
  `published_baselines`. The hybrid scores 0.524 under it.
- 2026-11-14: PFF as training data: +0.005 under `label_free`, -0.021 under `match_like`; PFF alone
  -0.016 and -0.082. On the PFF test matches such models score 0.10 to 0.14 higher, which does not
  carry over. Not adopted.
- 2026-11-14: version 3 is `tau2`, costs 8 and 20 (`configs/hybrid_bet.toml`), chosen on two matches
  and scored on the third. `possession decide`: mean +0.024, but -0.030 under `label_free` / `touch`
  and -0.052 on the PFF test matches (0 of 16). **Version 1 is submitted; the file is unchanged.**
- 2026-11-14: the hybrid with our label-free evidence at half weight: +0.022 under `label_free`,
  -0.021 under `match_like`, -0.012 on IDSSE, +0.009 on PFF. Stays out.
- 2026-11-14: `scripts/variant_grid.py` reproduces the two grids (training data, version 3 family).

## Left for the freeze (task 06)
- `make submit CONFIG=configs/hybrid_supervised.toml TEAM=culers` on a clean `main`; the sha256
  must be `25f146c0...`.
- Owner: confirm that the PFF terms allow this use (validation only; nothing trained on PFF is
  published). Owner: the form.
