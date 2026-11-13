---
status: todo
owner:
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
