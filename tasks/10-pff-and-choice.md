---
status: doing
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
- [ ] version 3 chosen from its family by nested selection and compared with version 1 by
      `possession decide`; the decision logged
- [ ] the addition that `published_baselines` alone had rejected measured under the evidence
      profiles and on IDSSE
- [x] kick-off rule audited against Smart Tagging

## Log
