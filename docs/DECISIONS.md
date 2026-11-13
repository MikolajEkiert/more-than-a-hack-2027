# Decisions

One line per decision, newest last: date, who, what, why. The technical report is written from
this file, so record what was tried and **dropped** too.

- 2026-11-13 · setup · Three-stage pipeline (ball_state / team / decode) with a method registry. Lets the team work in parallel without duplicating code; see docs/ARCHITECTURE.md.
- 2026-11-13 · setup · The public 5-minute GT is used as a sanity check only (RULES.md 2). Selection uses label-free diagnostics and other-game validation.
- 2026-11-13 · setup · Reference pipeline `configs/baseline_smooth50.toml` matches starter/baseline.py frame for frame (0 mismatches on 158,857 graded frames of a synthetic full-size ATD). Equivalence with the committed baseline_smooth50.csv on the official data is NOT yet verified: tests/test_baseline_equivalence.py runs once the data is in data/.
- 2026-11-13 · setup · The local PR job now also fixes review findings and merges: fix commits are pushed to the PR branch only after `make check` passes, and a PR is merged when the review verdict is "looks good" and `make check` passes on main + PR. No human approval step; label `no-auto` opts a PR out. The job rewrites the PR branch (adds fix commits, and merges main into it); pull before continuing on it. The fixer may not edit the gate itself (`scripts/`, `Makefile`, `pyproject.toml`, `AGENTS.md`, `.claude/`...), and a PR that touches organiser files or `scripts/protected.sha256` is never merged automatically; `check_repo.py` also fails when the manifest differs from origin/main unless `ALLOW_MANIFEST_CHANGE=1`. Chosen for speed with four days left; GitHub Actions cannot gate (billing lock).
