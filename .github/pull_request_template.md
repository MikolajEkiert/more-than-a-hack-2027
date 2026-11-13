## Task
tasks/NN-....md: <one line: what this PR does toward the task's goal>

## What changed
-

## Evidence
- [ ] `make check` passes locally
- [ ] experiment logged (`experiments/runs/...json`) if labels change: <name, poss count, mean dur>
- [ ] public 5-min window used as a sanity check only, not to choose between versions

## Rules (RULES.md)
- [ ] no hand labels or hand corrections of the provided game; no frame-specific constants
- [ ] no organiser files edited (`evaluation/`, `starter/*.py`); no data, caches or submissions committed
- [ ] deterministic (any randomness takes an explicit `seed`)
- [ ] new method registered in the right stage, with a docstring; no duplicate of an existing one
- [ ] docs/DECISIONS.md / docs/TECHNICAL_REPORT.md updated if the approach changed
