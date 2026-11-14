---
status: doing
owner: MikolajEkiert (Claude)
stage: docs
---

# 06 · Technical report, public repo, final submission

## Goal
Two of the three deliverables are not the CSV. Without them the score does not count.

## Scope
- May touch: `docs/TECHNICAL_REPORT.md`, `README.md` (team section on top), configs.

## Acceptance criteria
- [ ] report complete (`/report`), every number linked to a run file
- [ ] README: install (`uv sync`), data location, regenerate (`make submit ...`)
- [ ] `/submit` done on a clean `main`: reproducibility check passes, official validator OK
- [ ] repo public, `scripts/check_repo.py` green, no data in history (`git log --all -- data/`)
- [ ] form submitted before 2026-11-17 with `<team>.csv` and `<team>.pdf`/`.md`

## Log
- 2026-11-14 · MikolajEkiert/Claude · Fallback candidate prepared, not submitted (owner's decision:
  it waits on the branch `fallback/hybrid` and is sent only if nothing better is found). Done:
  technical report for `configs/hybrid_supervised.toml` (all seven sections), team section on top
  of the README with the one command that regenerates the file and a note that
  `python -m solution.submission` and the notebooks are not the submission path, the trained
  model committed in `models/` (loaded only when the package's cache key matches; same file with
  an empty cache in 11 s and with `MTAH_NO_CACHE=1` from scratch in about two minutes),
  `make submit ... TEAM=culers` on this tree: sha256 `25f146c0...`, official validator passes.
  Open: the same check on Windows; the final choice between versions (tasks 09-10); the form.
