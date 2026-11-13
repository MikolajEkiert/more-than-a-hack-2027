---
status: doing
owner: MikolajEkiert (Claude)
stage: decode
---

# 05 · Decode probabilities into whole possessions

## Goal
spellF1 scores possessions as intervals. A decoder that knows possessions last seconds, and that a
change of possession needs *control* (CHALLENGE.md), removes flicker without making transitions late.

## Context
`majority_vote` is a window vote: it lags transitions by about the window and ignores confidence.
Options: an HMM / Viterbi over {H, A, D} with switching costs, hysteresis with a minimum duration
of control before a hand-over, segment-level decisions, and the X policy (abstain inside an
occlusion = free; never across a hand-over). Needs soft inputs from 03 and 04.

## Scope
- May touch: `src/possession/stages/decode.py`, `features.py`, tests, configs.

## Interface
`decode.<name>(match, p_alive, p_home, ...)`, parameters in seconds.

## Acceptance criteria
- [ ] synthetic tests: a single-frame flip does not create a possession; a sustained change is
      placed within a few frames of its true time
- [ ] runs logged against `majority_vote` on the same inputs (count, duration, other-game validation)
- [ ] `make check` green

## Log
- 2026-11-13 · MikolajEkiert/Claude · `decode.switch_cost` added (Viterbi over H/A/D with switching
  costs, per half). Synthetic tests: a single-frame flip creates no possession; a sustained
  change lands within 3 frames; no `X` inside a half. Same inputs, external match:
  `unsup_control` 0.503 [0.466, 0.546] against 0.321 for `majority_vote` on the baseline stages.
  With perfect inputs it scores 0.994. The 0.503 is in-sample: the switching costs (and the team
  stage's control radius) were chosen on that same match; a held-out score is not measured. Not done: explicit durations, boundary rules across
  ball gaps.
