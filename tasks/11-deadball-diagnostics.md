---
status: done
owner: MikolajEkiert (Claude)
stage: ball_state, infra
---

# 11 · Where stoppages go wrong, and which ball regime the match is in

## Goal
Measure, with aggregates only, where the submitted hybrid gets stoppages (D) wrong on every
validation set, how much perfect stoppages would be worth, and whether the competition match
behaves like automatic tracking that loses the ball during stoppages. The answers decide whether
tasks 12 (training regime) and 13 (temporal model) go ahead (docs/FUSION.md section 10).

## Context
On the 16 held-out PFF matches, replacing only our p_alive with that of two other teams' models
(keeping our p_home and our decoder) raised spellF1 from 0.773 to 0.80-0.84, while our team
stage is the best of the three. The probability that the ball is missing given a stoppage is
0.99 on PFF and 0.87-0.96 on the degraded Metrica matches, but only 0.42 on IDSSE as degraded for
training (the package removes the ball without looking at the labels). Hypothesis: the model has
learned stoppages from where the ball is and how it moves, not from it disappearing.

## Scope
- May touch: `src/possession/inspect_data.py` (`deadball_parts`, `summarise_deadball`,
  `deadball_errors`, `regime`), `src/possession/cli.py` (`inspect regime`),
  `scripts/deadball_errors.py`, tests, docs.
- Out of scope: any change to a method or a configuration.

## Interface
- `possession inspect regime [--config C]`: on the competition match, the share of frames labelled
  D with a tracked ball and how often that ball moves at 4 m/s or more inside the pitch.
- `uv run python scripts/deadball_errors.py [SET ...] [--config C] [--compare DIR]`: the error
  split, the regime and the oracle bound (true stoppages, the configuration's own team evidence and
  decoder) on pff, idsse, metrica_label_free and metrica_match_like.

## Acceptance criteria
- [x] aggregate-only functions with tests on a synthetic match
- [x] error split, regime and oracle bound measured on the four sets
- [x] gates G1 (oracle bound) and G2 (regime of the match) evaluated and logged in
      docs/DECISIONS.md; the rule for tasks 12-13 written in docs/FUSION.md section 10

## Log
- 2026-11-14: oracle bound +0.070 / +0.072 on Metrica, +0.038 IDSSE, +0.100 PFF: G1 passes.
- 2026-11-14: the planned moving-ball statistic does not separate true from invented stoppages on the
  labelled sets; replaced by the running-players statistic before any task 12-13 number existed.
  The match loses the ball almost only in stoppages and keeps it in view in about a quarter of its
  stoppage time: G2 takes the f < 0.5 branch (only the mixed regime member; plain IDSSE veto).
- 2026-11-14 (review of PR #19): the gates are now written out in docs/FUSION.md section 10, which
  also says what was fixed before and what during this task. G1 had no threshold written down before
  measuring: against the noise band (±0.04), Metrica (+0.070, +0.072) and PFF (+0.100) clear it,
  IDSSE (+0.038) does not. G2 compared the match's tracked-ball stoppages (0.25) with references for
  all true stoppages; the tracked-ball references (`running_given_true_dead_tracked`,
  `running_given_true_live_tracked`) are still to be quoted. Added a CLI test for `inspect regime`.
