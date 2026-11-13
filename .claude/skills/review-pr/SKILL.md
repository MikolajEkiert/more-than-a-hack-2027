---
name: review-pr
description: Review a teammate's pull request for rule compliance, duplication, contract violations, determinism and correctness. Use when asked to review a PR (e.g. "/review-pr 12") or before merging.
---

# review-pr <number>

1. `gh pr view <n>` and `gh pr diff <n>`. Read the linked task file and `AGENTS.md`.
2. Check `gh pr checks <n>`. A red CI is the first finding.
3. Review against this list. Report only real problems, each with file:line and a concrete
   failure scenario, most severe first:
   - **Rules**: hand labels, frame-specific constants for this match, public GT used to tune or
     choose, human data about this game, data/caches/submissions/notebook outputs committed.
   - **Organiser files** touched (`evaluation/`, `starter/*.py`).
   - **Duplication**: logic that already exists in another method, `features.py`, `starter/` or
     `possession_metrics.py`. Name the existing one.
   - **Contract**: registered with a docstring and defaults; returns the right shape and range; NaN
     handled (ball missing on 41% of graded frames); no mutation of `Match`; frame indexing is
     video frame (ATD frame_id - 1).
   - **Determinism**: unseeded randomness, dict or set ordering, time-dependence.
   - **Scope**: changes outside the task's declared scope; shared contract changes not noted
     in docs/DECISIONS.md.
   - **Evidence**: the claims in the PR match the logged run; noise-band caveats stated.
4. Post the review: `gh pr review <n> --comment --body "<findings>"` (or `--approve` if clean and
   the user agrees). Never merge without the user's go-ahead.
