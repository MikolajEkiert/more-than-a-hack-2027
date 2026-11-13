---
name: submit
description: Produce the final, reproducibility-checked submission file and the final checklist for the Google Form. Use only when the user asks to prepare the submission.
---

# submit

1. Confirm with the user: which config, and the exact team name (the file must be `<team>.csv`).
2. Be on an up-to-date `main` with a clean tree (`git status`): the submitted file must come from
   a commit that is public.
3. `make check`, then `make submit CONFIG=configs/<x>.toml TEAM=<team>`. This runs the pipeline,
   re-runs it from a fresh parse, requires identical sha256, and runs the official
   `evaluation/validate_submission.py` on `submission/<team>.csv`.
4. Update the reproduction section of `docs/TECHNICAL_REPORT.md` with the command, config, commit and
   sha256. Export it to `<team>.pdf` or submit the `.md`.
5. Final checklist for the user:
   - [ ] repo is public and contains no data (`python3 scripts/check_repo.py` is green on main)
   - [ ] README tells how to install (`uv sync`) and regenerate (`make submit ...`)
   - [ ] the CSV uploaded is the one just validated (same sha256)
   - [ ] the form has the team name, the CSV, the repo URL and the report; deadline 2026-11-17
6. Never upload anything yourself.
