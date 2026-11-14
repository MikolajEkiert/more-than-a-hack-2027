---
status: review
owner: MikolajEkiert (Claude); reviewer: alargh
stage: ball_state, team
---

# 08 · The supervised model inside the pipeline

## Goal
Make the hybrid (`configs/hybrid_supervised.toml`: the supervised model's evidence, our kick-off
rule, our decoder) a submission candidate: the repository trains the model itself and regenerates
the same file, instead of reading probability files that exist on one machine.

## Context
docs/FUSION.md section 6: the hybrid is accepted by the rule against both single tracks under
both degradation profiles (0.659 / 0.561 against 0.621 / 0.506 and 0.598 / 0.448). The model lives
on the branch `supervised_approach` (package `solution/`), which starts from the organisers'
repository, not from `main`.

## Scope
- May touch: merge of `supervised_approach` (adds `solution/`, `notebooks/`,
  `requirements-solution.txt`); new `src/possession/supervised.py`; one method each in
  `stages/ball_state.py` and `stages/team.py`; `fusion.py` (shared conversion of three class
  probabilities); `pyproject.toml`, `uv.lock`; the config; tests; docs.
- In `solution/`: only where the open data comes from (`idsse.py`). Nothing that changes the
  features or the model.
- Out of scope: the package's own decoder, command line and kick-off rule (not called); changing
  the model.

## Interface
`ball_state.supervised(labels="raw")`, `team.supervised(labels="raw")`. `labels` is the label
function the model is trained on: `raw`, `tau<seconds>`, `controlled<seconds>`.

## Acceptance criteria
- [x] `make run CONFIG=configs/hybrid_supervised.toml` from an empty cache gives the same sha256
      as the hybrid built from the branch's probability files
- [x] external validation unchanged (0.659 / 0.561), runs logged
- [x] tests without data or network (`tests/test_supervised.py`, contract tests through a stub)
- [x] training data in `docs/TECHNICAL_REPORT.md` section 3; docs/DECISIONS.md lines
- [x] the same sha256 on other systems: a fresh clone, a Linux arm64 container, a Linux x86-64
      container under emulation
- [ ] the same sha256 on a teammate's computer (Windows)

## Log
- 2026-11-14 · MikolajEkiert/Claude · Merged the branch as written. Bridge
  `possession/supervised.py` (match layout, label functions by name, model trained once and
  cached by the package, probabilities memoised on the match) and the two methods. Open data in
  `.cache/idsse/`, package cache in `.cache/solution/`, nothing in `data/`. In
  `solution/idsse.py`: download folder by `MTAH_IDSSE_DIR`, fixed dataset revision, sha256 check
  of every download against `solution/idsse.sha256`. Result: from an empty cache, under this
  repository's lock, the model file is byte-identical to the branch's and the submission file has
  the sha256 it had before (`25f146c0...`); validation 0.659 / 0.561.
- 2026-11-14 · MikolajEkiert/Claude · Merged as PR #13. The same sha256 from an empty cache in a
  fresh clone (download included), in a Linux arm64 container and in a Linux x86-64 container
  under emulation. What is left: the same check on a teammate's Windows machine, then freezing
  (task 06).
