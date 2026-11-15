---
status: doing
owner: MikolajEkiert (Claude)
stage: all
---

# 20 · Integration of tasks 16-19, documentation, submission readiness

## Goal
Bring the four experiment branches (16-19) into one branch and build version 6 from whatever
passed rule D. Check that the submitted file is still reproduced byte for byte, document the
outcome in the technical report, and keep the comparison with other teams reproducible from the
repository.

## Scope
- May touch: merges of `task/16`-`task/19`, `docs/TECHNICAL_REPORT.md`, `docs/FUSION.md`
  (section numbering), `docs/DECISIONS.md`, `scripts/compare_teams.py` (new), this file.
- Must not touch: `solution/`, `sequence.py`, `sequence_train.py`, `regime.py`, organiser files,
  `submission/culers.csv`.

## Acceptance criteria
- [x] Branches 16, 19, 17 and 18 merged. Conflicts were in DECISIONS.md and FUSION.md, plus
      `supervised.py`, where tasks 17 (`data=`) and 18 (`tta=`) both changed `_team_folds`. The two
      options are combined, and the defaults keep version 4's keys.
- [x] Every verdict re-run by the integrator with `grid14.py decide-d --bins truth:c14_best` on
      the agents' cells. All four tasks keep version 4:
      - 16: PFF -0.0082;
      - 17: held out -0.0081;
      - 18: `touch` floor broken and PFF -0.087;
      - 19: +0.0011 against a threshold of +0.0054.
      No version 6.
- [x] `make check` green after the merges (221 tests).
- [x] Version 4 regenerated from the merged code: sha256 `6fd87cc4467530fa…`. Version 5: `0167a8c6c243f1fd…`.
- [x] FUSION.md sections renumbered (14 task 16, 15 task 17, 16 task 18).
- [x] TECHNICAL_REPORT section 5: a table of the four ideas and why each stays out; "tried and
      dropped" extended.
- [x] `scripts/compare_teams.py` (moved from a scratch script) measured and reported. The PFF
      comparison is in the report: V4 0.820, Barca 0.743, Sigue 0.767, 16/16 against each.
- [ ] Owner's decision between version 4 and version 5. Then push, PR (`no-auto`) and `/submit`
      from `main`.

## Log
- 2026-11-15: merged, verified, documented. The seed networks of task 19 stay uncommitted, since
  no t19 configuration is submitted.
