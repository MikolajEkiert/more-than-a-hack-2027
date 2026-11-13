# Tasks

One file per unit of work: `NN-short-slug.md`, created from `_TEMPLATE.md` (or with `/plan-task`).
A task file is the **shared memory** between people and agent sessions: any agent picking it up
reads it first and appends to its log before stopping.

- Branch `task/NN-slug`, one PR. The session hook shows the current branch's task automatically.
- `status:` todo → doing → review → done (or `dropped`, with the reason in the log).
- `owner:` one person. Do not start a task that someone else has `doing`.
- Acceptance criteria are checkable. "Public spellF1 goes up" is never one (RULES.md 2).

## Suggested split for 3 people (change freely)

| person | first | then |
|---|---|---|
| A: infra and evaluation | 01 data and loader, 02 validation without the public GT | 06 report and submission |
| B: dead ball | 03 `ball_state` detector | 05 decoder (with C) |
| C: team | 04 `team` under occlusion | 05 decoder (with B) |

Stages are independent files, so B and C rarely conflict. 05 combines both and starts once 03/04
have a first method each.
