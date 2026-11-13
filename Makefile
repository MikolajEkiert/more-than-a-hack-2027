# One entry point for humans, agents and CI. `make help` lists targets.
# Windows without make: run the commands after each target by hand.
CONFIG ?= configs/baseline_smooth50.toml
NOTE ?=

.PHONY: help setup inspect review-watch-install review-watch-uninstall review-watch-status check lint fmt test run methods log eval submit guard-on guard-off protect-update

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## install the pinned environment and the git hooks
	@case "$$PWD" in "$$HOME/Desktop"*|"$$HOME/Documents"*|*"Mobile Documents"*) \
	  echo "WARNING: repo is in a folder iCloud may sync (see docs/TEAM_SETUP.md: move it).";; esac
	uv sync
	uv run pre-commit install

check: lint test  ## definition of done: lint + tests + repo guard
	python3 scripts/check_repo.py

lint:  ## ruff lint and format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

fmt:  ## auto-format and auto-fix
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

test:  ## unit + contract tests (no match data needed)
	uv run pytest

run:  ## make run CONFIG=configs/x.toml NOTE="what this tests"
	uv run possession run $(CONFIG) --note "$(NOTE)"

methods:  ## registered methods per stage (check before writing a new one)
	uv run possession methods

log:  ## experiment log
	uv run possession log

inspect:  ## make inspect TOPIC=ball|players|tagging (aggregates only)
	uv run possession inspect $(or $(TOPIC),ball)

eval:  ## official scoreboard over evaluation/results/ (public 5 min: sanity only)
	uv run python evaluation/run_evaluation.py

submit:  ## make submit CONFIG=configs/x.toml TEAM=name
	@test -n "$(TEAM)" || (echo "TEAM=<team name> required" && exit 1)
	uv run possession submit $(CONFIG) --team $(TEAM)

guard-on:  ## data guard ON: agents may no longer read match data (official dataset)
	python3 -c "import json,pathlib;p=pathlib.Path('scripts/agent_guard.json');d=json.loads(p.read_text());d['enabled']=True;d['currently']='official dataset in data/: guard on';p.write_text(json.dumps(d,indent=2)+'\n')"
	printf 'data/\n.cache/\nsubmission/\nevaluation/results/*.csv\n!evaluation/results/baseline_*.csv\n' > .cursorignore
	@echo "guard ON. Commit scripts/agent_guard.json and .cursorignore so the whole team gets it."

guard-off:  ## data guard OFF (sample data only)
	python3 -c "import json,pathlib;p=pathlib.Path('scripts/agent_guard.json');d=json.loads(p.read_text());d['enabled']=False;d['currently']='sample data: guard off';p.write_text(json.dumps(d,indent=2)+'\n')"
	rm -f .cursorignore

review-watch-install:  ## local AI review, auto-fix and auto-merge of PRs every 5 min (ONE machine only; macOS launchd)
	scripts/review_watch_launchd.sh install

review-watch-uninstall:  ## stop the local PR reviewer / fixer / merger
	scripts/review_watch_launchd.sh uninstall

review-watch-status:  ## is the local PR reviewer running; last log lines
	scripts/review_watch_launchd.sh status

protect-update:  ## after deliberately pulling an organiser update of evaluation/ or starter/
	python3 scripts/check_repo.py --update-manifest
