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
3. PR → the review job (below) reviews, fixes and merges it; or a teammate runs `/review-pr <n>` and
   merges. `git pull` your branch before pushing again: the job may have added a fix commit.
4. `git pull --rebase` on main before starting the next task.

## Automatic AI review, fix and merge of PRs (one machine)
`make review-watch-install` runs `scripts/pr_review_watch.sh` every 5 minutes via launchd, from its
own clone. For each open, non-draft PR by a repo collaborator it does one step per head commit:

1. **Review** (Sonnet 5.5, effort high, read-only tools) following `/review-pr`; one comment from
   the `gh` account of that machine, starting with `Verdict: looks good | needs changes | blocking`.
2. **Fix**, if the verdict is not "looks good": a second session (Opus 5.5) edits a checkout of the
   PR branch. The script then runs `make check`, commits and pushes to the branch (never a
   force-push), and comments what was fixed, what was left and what needs a human. The new commit is
   reviewed on the next run. At most 3 fix rounds per PR.
3. **Merge**, if the verdict is "looks good": the script merges the PR into a scratch checkout of
   main, runs `make check` there, and only then merges on GitHub (a merge commit).

Controls:
- Label a PR **`no-auto`**, or keep it a draft, and it is not fixed or merged.
- `PR_REVIEW_AUTOFIX=0` gives review only. `PR_MERGE_METHOD=squash` changes the merge method.
- Test without posting, pushing or merging: `DRY_RUN=1 ONLY_PR=<n> scripts/pr_review_watch.sh`.
- Re-run `make review-watch-install` after changing the script or `AGENTS.md`: the job uses copies.

What to know: the fix step runs the PR's code (`make check`) on that machine, so it only handles
collaborators' branches in this repository, in a clone without the match data. Nobody approves a
merge; the gate is the reviewer's verdict plus `make check`. GitHub Actions is not part of the
gate (see `scripts/review_known_issues.md`).

State and worktrees are in `~/Library/Application Support/mtah-pr-review/`, the log is in
`~/Library/Logs/mtah-pr-review/`. Stop it with `make review-watch-uninstall`. The laptop must be
awake and logged in.

## When the official dataset arrives
`make guard-on`, commit `scripts/agent_guard.json` and `.cursorignore`, and tell everyone to pull.
From then on agents cannot read data/, .cache/, submissions or notebook outputs. The terms forbid
giving the dataset or extracts of it to a cloud LLM.
