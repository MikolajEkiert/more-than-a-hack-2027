---
name: experiment
description: Run, log and honestly interpret a pipeline experiment from a config in configs/. Use when asked to run, compare or evaluate a configuration, or to check whether a change helped.
---

# experiment

1. Pick or write the config in `configs/` (one TOML per experiment; the `name` becomes the CSV and
   the log entry). Unknown parameters fail loudly, which is the intended behaviour.
2. Run it: `make run CONFIG=configs/<x>.toml NOTE="<the hypothesis, one line>"`.
3. Compare with earlier runs: `uv run possession log`.
4. Interpret **without tuning on the public window**:
   - Label-free diagnostics first: possession count and mean duration (a football match has on the
     order of a couple of hundred possessions; hundreds of 2-second ones means flicker, a few
     dozen long ones means over-smoothing), home share, and the share of D and X.
   - Other-game validation, if a task has built it (labelled data from other games is allowed).
   - Public sanity spellF1/macroF1 only to confirm nothing is broken (a large drop means a bug,
     e.g. an off-by-one). Never pick between versions on it, and never sweep a parameter against it.
5. Write the conclusion into the relevant task file's log: hypothesis, numbers, verdict, next
   step. Commit the `experiments/runs/*.json` file with the code that produced it.
