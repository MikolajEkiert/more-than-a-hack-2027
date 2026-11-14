---
status: doing
owner: MikolajEkiert (Claude)
stage: infra
---

# 09 · A second validation set: the IDSSE matches, each scored by a model that never saw it

## Goal
Every choice so far was made on three Metrica matches. Add an independent set, so that a change
has to hold on another provider's matches too, and so that stoppages with the ball in view are
tested at all (Metrica's corrected tracking has no ball during stoppages; IDSSE's has).

## Context
The supervised model is trained on the seven IDSSE matches, so they can only be used out of
fold. The package's own harness (`solution/evaluate_external.run`) already trains one model per
held-out match and caches the predictions. Our decoder and kick-off rule were never tuned there.
docs/FUSION.md section 8 fixes what the set is for: a veto, not evidence.

## Scope
- May touch: `src/possession/supervised.py` (match conversion, held-out predictions, conditions),
  `src/possession/validation.py` (suites; label readings taken from the cells, not from a
  constant; stoppage context per cell), `src/possession/cli.py` and `Makefile` (`--suite`),
  tests, docs.
- Out of scope: any change to the model or to `solution/`.

## Interface
`possession validate CONFIG... --suite idsse [--against CONFIG]`. Readings scored:
`controlled1.5` and `controlled3.5` count (possession density 6.1 and 4.15 per live minute, like
the two Metrica readings; nobody's training labels); the provider's flags (`raw`, 8.3) are
reported only.

## Acceptance criteria
- [x] a training match can only be scored by a model that has not seen it: it carries its
      out-of-fold prediction, and without one `supervised.probabilities` raises (tested)
- [x] the truth for every reading is derived from the provider's flags, never from the labels a
      model was trained on
- [x] `summarise` / `compare` work with any reading names; readings that do not count are
      reported separately (tested)
- [x] Metrica numbers of the hybrid unchanged (0.659 / 0.561)
- [ ] first numbers on IDSSE logged: hybrid, `unsup_v2`, the package's own decoder; stoppages
      with the ball in view

## Log
