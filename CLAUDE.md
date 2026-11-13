@AGENTS.md

## Claude Code specifics

- Workflows are skills in `.claude/skills/` (shared with Cursor): `/plan-task`, `/do-task`,
  `/inspect-data`, `/experiment`, `/debug-run`, `/review-pr`, `/report`, `/submit`. Prefer them
  to ad-hoc procedures. Package rules: `src/possession/AGENTS.md` (loaded when working there).
- Hooks (`.claude/settings.json` -> `scripts/agent_hooks.py`) inject the session context (branch,
  current task, latest runs), block edits to organiser files, auto-format Python after edits and,
  while the data guard is on, block reading match data. A blocked call carries a reason. Follow
  it; do not work around it.
- When a task grows beyond its scope, stop and propose a new task file instead of expanding.
