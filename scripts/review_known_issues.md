# Known and accepted issues: the automated PR reviewer must NOT report these

Edit this list when the team accepts something; `make review-watch-install` copies it to the
reviewer (it is read from that trusted copy, never from the PR under review).

- **Red CI / failing `check` job.** GitHub Actions is blocked by a billing lock on the account; jobs
  fail in 2-3 s without running. This is not a code problem. Do not mention CI status at all.
  The local gate is `make check` + pre-commit.
- **`.claude/settings.json` missing** (Claude Code hooks: organiser-file protection, data guard,
  session context). It is pending the repo owner's approval and will come in a separate commit.
- **Data guard is off by the team's decision.** Gaps in the guard's heuristics (scripts/agent_hooks.py)
  are low priority while it is off; report only a clear bypass that is new in this PR.
- **Baseline equivalence on the official data is not verified yet** (tests/test_baseline_equivalence.py
  is skipped until someone runs it with the data). docs/DECISIONS.md says so.
