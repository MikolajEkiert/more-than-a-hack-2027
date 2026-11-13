---
status: doing
owner: MikolajEkiert (Claude)
stage: ball_state
---

# 03 · Detect dead ball properly

## Goal
`ball_visible` calls the ball dead whenever the tracker loses it, so D is noise, and every false
D splits a possession in two. A real stoppage detector removes the largest source of fake
possessions.

## Context
Signals available: the ball outside the pitch bounds (metres in `Match`), the ball missing for a
long stretch versus a short occlusion, collective player speed dropping at stoppages, players
regrouping for set pieces, and Smart Tagging `SET PIECES` intervals (restarts end a dead spell;
`Type` throw-in, corner, ...; note `Team` is N/A on throw-ins). FAQ: the second half's graded
window starts 2,934 frames of D before the kickoff.

## Scope
- May touch: `src/possession/stages/ball_state.py`, `features.py` (shared signals), tests, configs.
- Out of scope: team assignment, decoding.

## Interface
`ball_state.<name>(...)` returning soft `p_alive`, e.g. `stoppage_model(min_gap_s=..., ...)`.

## Acceptance criteria
- [ ] tests on synthetic cases: ball out of bounds becomes dead; short occlusion mid-play stays alive
- [ ] run logged against the baseline: D share and possession count plausible, reasoning written down
- [ ] no frame-specific constants; parameters in seconds or metres, not frame numbers of this match
- [ ] `make check` green

## Log
- 2026-11-13 · MikolajEkiert/Claude · `ball_state.stoppage_evidence` added. Synthetic tests: ball
  out of bounds lowers the evidence; no ball and nobody running is a stoppage; no ball with
  players sprinting is far less sure. External match (degraded): alive/dead F1 0.956 against
  0.906 for `ball_visible`, as part of `unsup_control`. On the match: D share 44.8%, 358
  possessions. Parameters are in seconds, metres and m/s; no frame constants. Not done: fitting
  the cue parameters on the match (EM), the kick-off rule before the second half.
- 2026-11-14 · MikolajEkiert/Claude · Kick-off rule added to `stoppage_evidence` (`features.kickoffs`,
  formation only). External matches: every kick-off event found, kick time +0.3 s. Our match:
  second-half kick-off 0.2 s from the published anchor; the graded wait before it is now 100% D
  (was 63%). A moving tracked ball inside the pitch now counts as evidence of play. Stoppage edges are snapped to the ball gap in the decoder (task 05). Not done:
  fitting the cue parameters on the match (a plain hard EM hurt on the external match).
