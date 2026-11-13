---
name: report
description: Update docs/TECHNICAL_REPORT.md (the submitted technical document) from DECISIONS.md, the task logs and the experiment log. Use when asked to write or update the report, or after a significant decision is merged.
---

# report

The report is a deliverable (SUBMISSION.md section 3). It is short: a few pages.

1. Read `docs/TECHNICAL_REPORT.md`, `docs/DECISIONS.md`, `tasks/*.md` (status done and review) and
   `uv run possession log`.
2. Update only the sections the new information affects. Keep the required sections:
   approach (what and *why*), assets used **and dropped**, training data (every labelled dataset:
   source, who labelled, how much, none from this game), possession rules implemented (loose
   balls, challenges, restarts), limitations, reproduction (exact commands, config, commit, sha256).
3. Every number cites its run file in `experiments/runs/`. Public-window numbers are labelled as a
   sanity check, never as performance.
4. Plain, specific English; no marketing. Show the diff to the user before committing.
