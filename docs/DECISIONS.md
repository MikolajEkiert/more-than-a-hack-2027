# Decisions

One line per decision, newest last: date, who, what, why. The technical report is written from
this file, so record what was tried and **dropped** too.

- 2026-11-13 · setup · Three-stage pipeline (ball_state / team / decode) with a method registry. Lets the team work in parallel without duplicating code; see docs/ARCHITECTURE.md.
- 2026-11-13 · setup · The public 5-minute GT is used as a sanity check only (RULES.md 2). Selection uses label-free diagnostics and other-game validation.
- 2026-11-13 · setup · Reference pipeline `configs/baseline_smooth50.toml` matches starter/baseline.py frame for frame (0 mismatches on 158,857 graded frames of a synthetic full-size ATD). Equivalence with the committed baseline_smooth50.csv on the official data is NOT yet verified: tests/test_baseline_equivalence.py runs once the data is in data/.
