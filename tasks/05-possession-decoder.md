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
  With perfect inputs it scores 0.994. Not done: explicit durations, boundary rules across
  ball gaps.
- 2026-11-14 · MikolajEkiert/Claude · Tie rule (switch at the first evidence of the new state, the
  same for both teams) and `snap_stoppage_edges`. `unsup_v2` against the control version on
  three matches, both label readings: +0.053 (`label_free`) and +0.050 (`published_baselines`),
  accepted by the rule under both. Costs chosen for the smallest loss across readings and
  profiles: turnover 5, stop 12, restart 12 nats. On the match: 326 possessions, mean 10.8 s,
  44.4% dead.
