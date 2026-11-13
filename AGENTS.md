# AGENTS.md

Single source of truth for every AI agent in this repository (Claude Code reads it via CLAUDE.md,
Cursor reads it natively). Humans: this is also the team's working agreement. Keep it short; put
detail in docs/ and link it.

## 1. The goal

More than a Hack 2027, Phase 1. For each of the **199,319 frames** of the tactical video of
Barcelona (H) vs Athletic (A), output `H`, `A`, `D` (dead ball) or `X` (abstain).

- **Ranked on `spellF1`**: possessions matched as whole intervals by temporal IoU. `macroF1` is the
  tie-break. Baseline to beat: **spellF1 0.2433** (`configs/baseline_smooth50.toml`).
- What wins is **getting possession boundaries right**: few false possessions, transitions on time
  (1 s late costs about 35%). Flicker is fatal; per-frame accuracy is not the target.
- `X` on a graded frame is forward-filled by the scorer: free inside a possession, expensive
  across a hand-over. Ungraded frames (before kickoff, halftime, after the whistle) are X.
- Deadline **2026-11-17**. Deliverables: the CSV, this public repo (must regenerate the CSV
  **byte for byte**), and a short technical document (`docs/TECHNICAL_REPORT.md`).

Read before designing anything: CHALLENGE.md (possession rules), EVALUATION.md (the metric),
DATA.md (assets, frame alignment, 41% missing ball). `evaluation/possession_metrics.py` is the
spec. Where a doc and that file disagree, the file is right.

## 2. Hard rules (breaking one can disqualify the team)

1. **No hand labels or hand corrections on the provided game**, in any amount. No constants tied to
   specific frames or moments of this match ("frames 51200-51340 are a throw-in"). Rules must
   be general enough to run on the next match.
2. **The public 5-minute ground truth is a sanity check only**: never training data, never a
   tuning set, never the reason to prefer version A over B (17 possessions; it picks the wrong
   window even for the baseline). It shows up as `public_sanity`. Tune on label-free diagnostics,
   football priors, Smart Tagging agreement, or labelled data from **other** games.
3. **No human-produced data about this game** (event data, corrected tracking) from any source.
4. **Deterministic.** Same repo + same data gives the same bytes. Randomness only via an explicit
   `seed` parameter. `make submit` proves it by running twice.
5. **Never commit data**: `data/`, `.cache/`, `submission/`, non-baseline `evaluation/results/*.csv`,
   notebook outputs, video. `scripts/check_repo.py` enforces this in pre-commit and CI.
6. **Do not edit organiser files**: `evaluation/*.py`, `evaluation/gt/`, committed baselines,
   `starter/*.py`. Import them; do not copy them. (Hooks block it; CI checks hashes.) `games.py` may
   be extended, e.g. to add a game.
7. **Data guard** (`scripts/agent_guard.json`). While it is ON, never read, grep, cat or print the
   match data, the `.cache/` arrays, submission CSVs or notebook outputs, and never write code
   whose output shows raw rows. Work through tools that print aggregates (`possession run`,
   `pytest`). It is OFF while we use the public sample data and goes ON for the official dataset.
   The terms forbid giving it to a cloud LLM.

## 3. Repository map

```
AGENTS.md            you are here         tasks/            one file per task (the backlog)
src/possession/      OUR CODE             configs/          one TOML per experiment / submission
  registry.py        method registry      experiments/runs/ one JSON per run (auto, committed)
  match.py           Match dataclass      docs/             ARCHITECTURE, DECISIONS, TECHNICAL_REPORT
  loader.py          data -> Match (+cache) tests/          synthetic; contract tests cover all methods
  features.py        shared derived quantities
  stages/            ball_state.py, team.py, decode.py (the methods)
  pipeline.py        config + Match -> labels   scoring.py  validation, diagnostics, sanity
  inspect_data.py    aggregate-only data stats   cli.py      `uv run possession ...`
  AGENTS.md          package rules (contract, conventions): read before editing here
evaluation/  starter/  games.py   ORGANISER CODE (read-only except games.py)
scripts/             check_repo.py, agent_hooks.py (hooks for both editors), agent_guard.json
```

## 4. Architecture in one paragraph (detail: docs/ARCHITECTURE.md)

`Match --ball_state--> p_alive --team--> p_home --decode--> labels`. Each stage is a set of
**registered methods** with a fixed contract. **A new idea is a new method, not a new pipeline.**
A config picks one method per stage and its parameters. Shared computations go in `features.py`.
Methods are pure: they read `Match` and never mutate it.

## 5. Before writing code (anti-duplication)

1. `uv run possession methods` lists what exists. Extend or parameterise an existing method
   before adding a near-copy.
2. Search `features.py`, `starter/` and `evaluation/possession_metrics.py` for the helper you need.
3. Read your task file in `tasks/`. Check that no other task in `status: doing` owns the same thing.
4. If the change touches a shared contract (`match.py`, `registry.py`, `pipeline.py`, `config.py`),
   keep it small and in its own PR, and note it in docs/DECISIONS.md.

## 6. Commands

```
make setup                   # once: uv sync (Python 3.11, pinned) + git hooks
make check                   # DEFINITION OF DONE: ruff + pytest + repo guard
uv run possession methods    # what is implemented
uv run possession inspect ball|players|gaps|tagging        # understand the data: aggregates only
uv run possession inspect labels --csv <file.csv>     # what a submission claims (no data needed)
make run CONFIG=configs/x.toml NOTE="what this tests"   # produce + validate + log a run
make log                     # experiment table
make eval                    # official scoreboard (public window: sanity only)
make submit CONFIG=... TEAM=<team>   # final file in submission/, reproducibility-checked
```

## 7. Workflow

Skills (shared by Claude Code and Cursor, in `.claude/skills/`): `/plan-task` (idea → task file),
`/do-task NN`, `/inspect-data` (understand the data safely), `/experiment`, `/debug-run` (numbers
look wrong), `/review-pr N`, `/report`, `/submit`.

- **One task = one file in `tasks/` = one branch `task/NN-slug` = one PR.** The task file holds the
  goal, scope (files you may touch), acceptance criteria and a log. Update its `status:` and
  append to its log as you go. That file is how the next agent session picks up the context.
- `No module named 'possession'` on macOS means the repo sits in an iCloud-synced folder, which
  hides `.venv/*.pth`; tell the human (docs/TEAM_SETUP.md). Do not patch the venv.
- Small PRs to `main`; `make check` must be green; another teammate (or `/review-pr`) reviews. On one
  machine a job reviews every PR, pushes fixes for its findings to the PR branch and merges it once
  the review is clean and `make check` passes on main + PR (docs/TEAM_SETUP.md). Label a PR
  `no-auto` to keep it manual; pull before you continue on a branch the job may have fixed. `git pull
  --rebase` often. Never push to `main` directly or force-push.
- Every run that changes labels: `make run ... NOTE=...` writes `experiments/runs/*.json`. Commit
  it with the code that produced it. Do not re-try ideas that the log shows already failed.
- A decision about the approach (a possession rule, a dropped asset) goes in a line in
  docs/DECISIONS.md, because the technical report is written from it.

## 8. Coding conventions

Python 3.11, numpy-first, vectorised over frames (199k frames: no per-frame Python loops in
methods). Type hints on public functions, docstring first line = what it does (shown by
`possession methods`). Arrays are indexed by **video frame** (ATD frame_id - 1). Distances in
metres (`Match` is already scaled). Handle NaN everywhere: the ball is missing on 41% of graded
frames, and players drop out. Every new method gets a focused test beyond the automatic contract
tests. No prints in library code except the CLI.

## 9. Definition of done

`make check` is green · the task's acceptance criteria are met and ticked in the task file · a run
is logged if labels changed · docs/DECISIONS.md is updated if the approach changed · the PR uses the
template. Report honestly: what was measured, what was not, and where differences are below the
noise band (spellF1 ±0.04).
