# Team setup (10 minutes, once per person)

## Everyone

**Clone outside iCloud-synced folders** (on a Mac with "Desktop & Documents" in iCloud, not in
`~/Desktop` or `~/Documents`; use e.g. `~/code/`). iCloud marks `.venv/*.pth` as hidden within
seconds, Python then skips them, and `uv run possession` fails with `No module named
'possession'`. It would also upload `data/` and `.cache/` (the match data) to iCloud.

```bash
git clone <repo> && cd more-than-a-hack-2027
# uv: https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)
make setup          # Python 3.11 + pinned deps + pre-commit hooks
make check          # should be green
# put the dataset in ./data (never committed) or export MTAH_DATA_DIR=/path
make run            # baseline through our pipeline
```
`gh auth login` once, so agents can open and review PRs.
Hooks call `python3`. On Windows, use WSL or make sure `python3` is on PATH.

## Claude Code (CLI or the VS Code extension)
Nothing extra: `CLAUDE.md` imports `AGENTS.md` (and `src/possession/CLAUDE.md` imports the package
rules in `src/possession/AGENTS.md` when Claude reads files there), `.claude/settings.json`
registers the hooks and `.claude/skills/` the workflows. Accept the project settings and hooks when asked the first time.
Start a session with `/do-task NN` or `/plan-task <idea>`.

## Cursor
Nothing extra: Cursor reads `AGENTS.md`, the nested `src/possession/AGENTS.md`,
`.cursor/rules/organiser-files.mdc` (organiser files are read-only), `.cursor/hooks.json`, and loads
the skills from `.claude/skills/`. If hooks do not fire, restart Cursor (Settings → Hooks shows
them). Invoke the same workflows (`/do-task NN`, ...).

## Daily loop
1. Pick a task in `tasks/` (or `/plan-task` a new one). Branch `task/NN-slug`.
2. Agent works via `/do-task NN`, and `make check` must be green.
3. PR → CI green → a teammate runs `/review-pr <n>` → merge.
4. `git pull --rebase` on main before starting the next task.

## When the official dataset arrives
`make guard-on`, commit `scripts/agent_guard.json` and `.cursorignore`, and tell everyone to pull.
From then on agents cannot read data/, .cache/, submissions or notebook outputs. The terms forbid
giving the dataset or extracts of it to a cloud LLM.
