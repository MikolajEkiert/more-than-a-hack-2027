---
name: do-task
description: Implement one task from tasks/ end to end on its own branch: claim it, check for existing code, write tests, implement, verify with make check, log a run, update docs, open a PR. Use when asked to work on, do, implement or continue a task (e.g. "/do-task 03").
---

# do-task NN

## 0. Orient (always, even when resuming)
1. Read `AGENTS.md` and `tasks/NN-*.md` fully, including its log: it is the memory of earlier
   sessions. Read `docs/ARCHITECTURE.md` if you will touch `src/possession/`.
2. `git fetch && git status`. If the branch `task/NN-<slug>` exists, switch to it and
   `git pull --rebase origin main`. Otherwise, from an up-to-date `main`:
   `git switch -c task/NN-<slug>`.
3. Set `status: doing` and `owner: <git user.name>` in the task file if not set.

## 1. Before writing code
- `uv run possession methods` and a search of `src/possession/features.py`, `starter/`,
  `evaluation/possession_metrics.py` for what you need. Reuse; do not re-implement.
- Write the plan as a short checklist in the task file's log (3 to 7 items). If the task is bigger
  than its scope, stop and tell the user; propose splitting with `/plan-task`.

## 2. Implement
- Test first where practical: a focused test in `tests/` on the synthetic match (`tests/conftest.py`
  `make_match`) that pins the behaviour you intend. The contract tests cover shape, range and
  determinism automatically once the method is registered.
- Implement as a registered method (`@register(stage, name)`, docstring, defaults for every
  parameter), vectorised and NaN-safe. Shared helpers go in `features.py`.
- Add or update a config in `configs/` that uses the method.
- Never edit organiser files. While the data guard is on, never read data or print rows.

## 3. Verify
- `make check` must be green. Fix the cause; never skip, weaken or delete a test to get there.
- If data is available: `make run CONFIG=configs/<x>.toml NOTE="<what this tests>"`. Report
  possession count, mean duration and label shares. Compare with the baseline run on these
  label-free numbers. Mention the public sanity spellF1 only as a sanity check, and note that
  differences under ±0.04 are noise.

## 4. Finish
- Tick the acceptance criteria in the task file. Append a log entry: what was done, the numbers
  observed, open issues. Set `status: review`.
- One line in `docs/DECISIONS.md` if the approach or a possession rule changed.
- Commit (code + test + config + `experiments/runs/*.json` + task file), then
  `git push -u origin task/NN-<slug>` and
  `gh pr create --fill --body-file .github/pull_request_template.md` (edit the body to fill it in).
- Tell the user the PR link and anything that needs a human decision.
