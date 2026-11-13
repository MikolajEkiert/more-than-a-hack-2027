---
status: todo
owner:
stage: infra
---

# 01 · Data in, loader verified, cache working

## Goal
Every teammate can `make run` on real tracking: the sample dataset now, the official one later.
Without this, no experiment produces a number.

## Context
`src/possession/loader.py` parses the EPTS ATD (`<frame>:<slots>:<ball>`) into `Match` and caches
it in `.cache/`. It is verified on synthetic data, where the pipeline equals `starter/baseline.py`
frame for frame. `games.py` holds per-game constants and paths.

## Scope
- May touch: `games.py` (add the sample game's entry), `src/possession/loader.py`, tests, configs.
- Out of scope: any method in `stages/`.

## Acceptance criteria
- [ ] sample dataset unpacked locally (not committed); if its layout or format differs, `games.py`
      and the loader handle it (adapter, not a fork of the loader)
- [ ] `make run CONFIG=configs/baseline_smooth50.toml` works on it; parse time and cache hit noted
- [ ] when the official data arrives: `uv run pytest -m needs_data` passes (baseline equivalence)
      and `make guard-on` is committed
- [ ] `make check` green

## Log
