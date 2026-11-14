# Known and accepted issues: the automated PR reviewer must NOT report these

Edit this list when the team accepts something; `make review-watch-install` copies it to the
reviewer (it is read from that trusted copy, never from the PR under review).

- **Red CI / failing `check` job.** GitHub Actions is blocked by a billing lock on the account; jobs
  fail in 2-3 s without running. This is not a code problem. Do not mention CI status at all.
  The local gate is `make check` + pre-commit.
- **`.claude/settings.json` missing** (Claude Code hooks: organiser-file protection, data guard,
  session context). It is pending the repo owner's approval and will come in a separate commit.
- **Data guard is off by the owner's decision**, also now that the official dataset is in `data/`
  (docs/DECISIONS.md, 2026-11-14; AGENTS.md hard rule 7 was reworded to "aggregates only"). Do
  not report the guard being off, and treat gaps in its heuristics (scripts/agent_hooks.py) as
  low priority. What is still a finding: code, a committed file or a logged run that prints or
  stores raw rows of the match data.
- **`tests/test_baseline_equivalence.py` is skipped where the match data is absent.** It was
  verified on the data on 2026-11-13 (docs/DECISIONS.md).
- **`solution/` and `notebooks/` are the supervised track's package, merged as written** (owner's
  decision, docs/DECISIONS.md 2026-11-14 task 08). Do not report style, structure, duplication or
  missing-test findings there. The pipeline calls only its features and model
  (`src/possession/supervised.py`); its own decoder, command line and kick-off rule, which reads
  the organisers' published kick-off frame, are not called and are not part of the submission.
  A correctness bug that changes the submitted file is still worth reporting. **Never edit
  `solution/` in a fix round**: its source is hashed into the model cache, so an edit retrains a
  different model and changes the submission; leave such findings to the team.
