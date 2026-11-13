---
status: todo
owner:
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
