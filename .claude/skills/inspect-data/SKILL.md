---
name: inspect-data
description: Understand the data or a signal through aggregate statistics before designing or changing a method (ball tracking quality, player counts, tagging coverage, what a submission looks like). Use when asked what the data looks like, why a signal fails, or before writing a detector.
---

# inspect-data

The data is never shown to you. `uv run possession inspect <topic>` prints aggregates only (rates,
percentiles, histograms, counts), so it is allowed even with the data guard on. Add `--json` for a
machine-readable dict, `--game G` for another game.

| Question | Command |
| --- | --- |
| How often is the ball tracked, how long are the gaps, is it speed-plausible, is it off the pitch? | `inspect ball` |
| Are about 22 players tracked, how fast do they move, how close is the nearest one to the ball? | `inspect players` |
| What does Smart Tagging cover, how many set pieces, is the ball tracked during them? | `inspect tagging` |
| What does a submission claim: label shares, number and length of possessions, flicker? | `inspect labels --csv PATH` |

`inspect labels` needs only the CSV, not the match data.

## Rules
- Never read `data/` or `.cache/`, never print rows, never run `python -c` that loads data.
- If the statistic you need is missing, add it as an aggregate in `src/possession/inspect_data.py`
  with a test in `tests/test_inspect.py`. Do not write an ad-hoc script. An aggregate must not
  expose a frame index, a per-frame value or the coordinates of a specific frame.
- Write the finding (numbers and what you conclude from them) into the log of the current
  `tasks/NN-*.md`.
- A finding informs a method's design or a parameter's range. It must never become a
  frame-specific constant or a lookup of this game's events (RULES.md 1): the method has to work
  on an unseen game.
- Compare numbers with football knowledge (22 players, about 25 minutes of ball-in-play per
  half, a possession lasting seconds, not tenths of a second), not with the public window.
