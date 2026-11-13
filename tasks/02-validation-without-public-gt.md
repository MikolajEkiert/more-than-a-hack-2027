---
status: review
owner: MikolajEkiert (Claude)
stage: infra
---

# 02 · A way to choose between versions that is not the public window

## Goal
Make decisions without tuning on the 17 public possessions (RULES.md 2, EVALUATION.md). This is what
stops the team from optimising noise.

## Context
Allowed sources of evidence: label-free diagnostics (`scoring.diagnostics`), football priors
(typical possession counts and durations), agreement with Smart Tagging (automatic, not ground truth),
and **labelled data from other games** (any amount, RULES.md), e.g. open tracking and event data
from which possession labels are derived by an algorithm.

## Scope
- May touch: `src/possession/scoring.py`, a new `src/possession/validation.py`, `games.py` (other
  games), tests, docs.
- Out of scope: methods in `stages/`.

## Interface
`possession run` reports the new evidence next to the diagnostics and logs it in the run JSON.

## Acceptance criteria
- [x] at least one validation signal beyond the diagnostics, documented in docs/ARCHITECTURE.md
- [x] the baseline windows 25/50/125 compared on it; does it prefer 50 over 125, as the full match
      does (EVALUATION.md)? That is the test of the signal.
- [x] any external dataset recorded in docs/TECHNICAL_REPORT.md §3 (source, who labelled, how much)
- [x] `make check` green

## Log
- 2026-11-13 · MikolajEkiert/Claude · `src/possession/validation.py` + `possession validate` /
  `make validate`. External match: Metrica Sample Game 3 (386 possessions, mean 10.1 s, 31.9% dead).
  Degraded (seed 0): baseline window 25 / 50 / 125 = spellF1 0.272 / 0.318 / 0.297, intervals
  [0.252, 0.292] / [0.292, 0.347] / [0.264, 0.330]. Same order as the full match, but only 25 < 50
  is established; 50 vs 125 is inside the noise band and not resolved. (`--clean` runs were not
  logged, so their figures are not quoted here.)
  Not measured: the spread over degradation seeds; how well the degradation matches the real
  tracker beyond the aggregates listed in DECISIONS. Next: every method PR reports `make validate`
  against `baseline_smooth50`.
- 2026-11-13 · review fixes · Scope deviations: besides the "May touch" list this PR also changes
  `cli.py` (new `validate` command), `experiments.py` (`external` field and table column), the
  Makefile (`make validate`), `AGENTS.md` and two configs (`vote_smooth25/125`). The interface
  line asked for the evidence in `possession run`; it is a separate `possession validate` command
  instead, so the competition run never needs the external download.
- 2026-11-13 · MikolajEkiert/Claude · Validation v2 (owner's approved plan). Three matches (Sample
  Games 1-3), two label readings (`touch`, `controlled`), two degradation profiles as files
  (`configs/degradation/label_free.json`, `published_baselines.json`), `possession calibrate`,
  `possession validate --against` with the acceptance rule. Numbers (mean spellF1 over 3 matches
  x 3 seeds x 2 readings): `label_free`: window 25 / 50 / 125 = 0.261 / 0.334 / 0.322, control
  0.522. `published_baselines`: 0.176 / 0.238 / 0.253, control 0.383. Window 50 beats 125 on the
  mean under `label_free` (as on the full match) but not under `published_baselines`; the two
  differ by 0.01-0.02 either way, inside the noise. Not measured: any profile's realism beyond
  the aggregates and published scores it was set from.

