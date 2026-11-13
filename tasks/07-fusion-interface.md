---
status: review
owner: MikolajEkiert (Claude)
stage: infra
---

# 07 · Fusion interface for a supervised track

## Goal
A teammate is building a supervised model. Make it possible to combine it with the label-free
pipeline as evidence in one decoder (not by voting on labels, which brings flicker back), and fix
in advance how the team chooses between fusion and one of the two outputs.

## Context
Both tracks can deliver the stage quantities the decoder already takes: `p_alive` and `p_home`
(docs/ARCHITECTURE.md). The supervised model's form is not known yet (registered methods, or
files of per-frame probabilities), so the interface accepts both. Validation is on the external
matches (task 02); the acceptance rule is `validation.compare`.

## Scope
- May touch: new `src/possession/fusion.py`, `src/possession/stages/ball_state.py` and `team.py`
  (one `blend` method each), `src/possession/validation.py` (condition name on degraded matches),
  `src/possession/cli.py` (`export`), `docs/FUSION.md`, a config, tests.
- Out of scope: the supervised model itself; choosing fusion weights (needs the model).

## Interface
`ball_state.blend(sources=[...], weights=[...])`, `team.blend(...)`; a source is a method name, a
table `{method=..., <params>}` or `{file="path/{game}.npz"}`. `possession export` writes the
tracking of every validation condition for a model kept outside the repo.

## Acceptance criteria
- [x] focused tests in `tests/test_fusion.py`, `make check` green
- [x] an example config that runs end to end and is scored on the external matches
- [x] the decision protocol written down before any supervised result exists (docs/FUSION.md)
- [x] docs/DECISIONS.md line

## Log
- 2026-11-14 · MikolajEkiert/Claude · `fusion.py` (weighted sum of log-odds; NaN = no vote; file
  reader for .npy/.npz incl. three class probabilities), `blend` methods, `possession export`,
  docs/FUSION.md. Example: blending our two team methods scores 0.589 against 0.598 for
  `touch_episodes` alone on the external matches (rejected by the rule): correlated sources add
  nothing. Next: the supervised track delivers methods or files; then steps 1-5 of FUSION.md.
