---
status: review
owner: MikolajEkiert (Claude)
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
- [x] sample dataset unpacked locally (not committed); if its layout or format differs, `games.py`
      and the loader handle it (adapter, not a fork of the loader)
- [x] `make run CONFIG=configs/baseline_smooth50.toml` works on it; parse time and cache hit noted
- [ ] when the official data arrives: `uv run pytest -m needs_data` passes (baseline equivalence)
      and `make guard-on` is committed
- [x] `make check` green

## Log
- 2026-11-13 · MikolajEkiert/Claude · Data moved into `data/tactical/` and `data/broadcast/` (the
  layout `games.py` expects; it was unpacked flat). `make run` on the baseline: parse + run 2.7 s,
  later runs read `.cache/`. `uv run pytest -m needs_data` passes, so the pipeline baseline equals
  the committed `baseline_smooth50.csv` on this data. Public sanity 0.3256, the figure
  EVALUATION.md gives for that baseline. The guard stays off by the owner's decision, so the
  third criterion's `make guard-on` is not done.
- 2026-11-13 · same · Scope extended by the owner's approved plan: shared features for the
  unsupervised methods (`features.ball_kinematics`, `ball_edge_distance`, `collective_speed`,
  `ball_gaps`, `runs`) and matching aggregates in the new `inspect gaps` topic. Findings on this data: ball gaps
  are 17 under 1 s, none between 1 and 3 s, 133 of 3 s or more (40.7% of graded frames); the ATD
  ball speed column is filled on 7,377 of 94,214 tracked frames and disagrees with the positions
  (median 1.3 vs 5.1 m/s). Next: tasks 02-05 per docs/research/UNSUPERVISED_RESEARCH.md.
