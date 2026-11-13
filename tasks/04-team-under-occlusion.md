---
status: doing
owner: MikolajEkiert (Claude)
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
- 2026-11-13 · MikolajEkiert/Claude · `team.control_evidence` added (evidence only when a player is
  within 2 m of a tracked ball; weak when the ball is passing by; none in a duel). Coverage on
  the external match: evidence on 40% of live frames, right team on 92% of those; team accuracy
  of the full control pipeline 0.863 against 0.821 for the baseline (in-sample: the 2 m radius
  was chosen on that match). This is the stage that limits
  the score (see DECISIONS). Not done: any evidence while the ball is not tracked (the decoder
  keeps the last owner), which is this task's original goal.
