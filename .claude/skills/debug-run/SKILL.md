---
name: debug-run
description: Diagnose a pipeline run whose numbers look wrong (sanity score collapsed, absurd possession count, validator failure, CSV not reproducible). Use when a run or submission looks broken, before changing any method.
---

# debug-run

Aggregates only. While the data guard is on (`scripts/agent_guard.json`) never read `data/`,
`.cache/` or CSVs and never print raw rows. Never tune on the public 5-minute GT: it is a bug
detector (a collapse means a bug), not a target. Never "fix" by editing organiser files.

Tools: `uv run possession log`, `uv run possession methods`, `make eval`,
`uv run python evaluation/validate_submission.py <csv>`,
`uv run possession inspect ball|players|tagging` (match-data aggregates),
`uv run possession inspect labels --csv <path>` (possession count, duration percentiles,
short-possession count, transitions/min, label shares per half).

Start: `uv run possession log` (when did it change? which config?), then `inspect labels` on the
bad CSV and on the baseline. Go down the list; stop at the first item with evidence.

1. **Validator failure / row count.** ~164k rows = broadcast indexed, not the tactical video
   (DATA.md). Need exactly 199,319 rows. Check: validator output, `inspect labels`. Fix: index
   by video frame; check first/last row for off-by-one.
2. **Frame off-by-one.** ATD `frame_id` is 1-based, video frame = `frame_id - 1`. A shift costs
   little macroF1 but moves every transition (spellF1 drops). Check: compare with the baseline's
   transition timing in aggregate; grep the code for `frame_id`. Fix: convert once, in loader.
3. **Flicker.** Hundreds of short possessions, high transitions/min, many short spells: decode
   problem (EVALUATION.md, `baseline_raw` spellF1 0.05). Fix: hysteresis / minimum duration /
   HMM in the decoder, not in the stage inputs.
4. **Over-smoothing.** Few long possessions, transitions late (1 s late costs ~35% spellF1).
   Fix: shorten the window or minimum duration; check the decode lag.
5. **H/A swapped.** Home share far from ~50% or sanity macroF1 below chance. Check
   `team_of_track` (home = first team in ATD metadata) and `inspect players`. Fix in loader.
6. **False D splitting possessions.** Many D runs inside play, D share too high: `ball_visible`
   treats every tracker loss as dead ball (ball missing on 41% of graded frames). Check
   `inspect ball`. Fix: p_alive must be uncertain, not 0, when the ball is merely missing.
7. **X policy.** X is forward-filled by spellF1: free inside a possession, but across a
   hand-over it carries the old label. Check X share and where X runs sit next to transitions.
8. **Second-half trap.** Graded from 110,808 but kickoff at 113,742: those 2,934 graded frames
   are D. Check label shares in half 2 (`inspect labels`). Fix generally (kickoff detection),
   never with a frame constant.
9. **NaN propagation.** NaN reaching decode, or a score that is NaN/constant. Check `inspect ball`
   NaN shares; the contract requires no NaN in `p_alive` and NaN = no evidence in `p_home`.
   Fix: masks and nan-aware reductions inside the method.
10. **Non-determinism** (`make submit` sha mismatch). Suspects: unseeded RNG (add `seed`), set or
    dict ordering, threads, stale `.cache/` (key = size + mtime; run with `--rebuild-cache`).
    Run the same config twice and compare shas.
11. **Contract violation.** `pipeline.py` raises on shape, range, dtype, unknown label or unknown
    param. Read the message; fix the method, not the pipeline. Run `make check`.

Finish: write the diagnosis (symptom, evidence as aggregates, fix or what was ruled out) in the
log of the relevant `tasks/NN-*.md`. Report the root cause with evidence, or say plainly what was
ruled out and what remains unknown. If a shared contract must change, propose a task.
