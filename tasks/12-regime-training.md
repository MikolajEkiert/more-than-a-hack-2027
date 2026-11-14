---
status: review
owner: MikolajEkiert (Claude)
stage: ball_state, team, infra
---

# 12 · Ball-state evidence from a model trained in the automatic-tracking ball regime, and test-time augmentation

## Goal
Measure members M3 and M4 of the family in docs/FUSION.md section 10 and decide by its rule.

## Context
Task 11: the IDSSE training matches keep the ball during stoppages and the package removes it at
random, while automatic tracking (PFF, Metrica, the competition match) loses it almost only during
stoppages. The competition match also keeps the ball in view during about a quarter of its stoppage
time, so only the mixed member (noise seed 0 as before, seed 1 in the automatic regime) is run.

## Scope
- May touch: `src/possession/regime.py` (new), `src/possession/supervised.py` (regime training,
  fold loop that keeps its models, test-time augmentation, the `idsse_auto` inputs),
  `stages/ball_state.py` and `stages/team.py` (parameters `regime`, `tta`), `validation.py` and
  `cli.py` (suite `idsse_auto`), configs, `scripts/family_grid.py`, tests, docs.
- Out of scope: `solution/` (never edited); the temporal model (task 13).

## Interface
- `ball_state.supervised(labels, data, regime, tta)`, `team.supervised(...)`; `regime` is "none",
  "auto_lf", "auto_ml" or "mix_lf"; `tta` averages in the home/away-swapped view.
- `possession validate ... --suite idsse_auto`: the IDSSE matches in the automatic-ball regime of
  `match_like`, out of fold (reported only).
- `uv run python scripts/family_grid.py`: every member on every set, the nested choice and the rule.

## Acceptance criteria
- [x] the regime transform never touches labels or players and is deterministic (tests)
- [x] every new learned part is out of fold on IDSSE; the fold loop reproduces the package's
      harness bit for bit for the plain regime (runtime check)
- [x] M3 and M4 measured on all sets; G3 checked; results logged (the decision is taken over the
      whole family with task 13)

## Log
- 2026-11-14: M3 +0.009 / -0.001 on Metrica, 0.000 IDSSE (2/7), +0.018 PFF; M4 within +-0.003
  everywhere. G3 fails (PFF +0.018, invented stoppages with the ball tracked -28%).
