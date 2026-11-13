---
status: todo
owner:
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
- [ ] at least one validation signal beyond the diagnostics, documented in docs/ARCHITECTURE.md
- [ ] the baseline windows 25/50/125 compared on it; does it prefer 50 over 125, as the full match
      does (EVALUATION.md)? That is the test of the signal.
- [ ] any external dataset recorded in docs/TECHNICAL_REPORT.md §3 (source, who labelled, how much)
- [ ] `make check` green

## Log
