---
status: todo
owner:
stage: team
---

# 04 · Who has the ball when the ball is not tracked

## Goal
On 41% of graded frames there is no ball, so `nearest_player` has no evidence (NaN). Inferring the
team from what is visible improves hand-over timing, which is what spellF1 punishes.

## Context
Ideas: carry the last confident team through short gaps (which interacts with decode: X is
forward-filled), ball interpolation between sightings, team shape and direction of movement
(the attacking team advances, the defending team retreats and compresses), pressure (opponents
converging), the closest player's distance as a confidence, Smart Tagging `Team` on attacking phases
as a weak prior.

## Scope
- May touch: `src/possession/stages/team.py`, `features.py`, tests, configs.
- Out of scope: dead-ball detection, final decoding.

## Interface
`team.<name>(...)` returning soft `p_home` (use values between 0 and 1 to express confidence;
NaN only where there is truly no evidence).

## Acceptance criteria
- [ ] tests on synthetic cases (a clear carrier gives a confident p_home; ambiguous cases sit near 0.5)
- [ ] coverage reported: share of alive frames with evidence, before and after
- [ ] run logged; `make check` green

## Log
