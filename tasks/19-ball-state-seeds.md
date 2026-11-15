---
status: doing
owner: MikolajEkiert (Claude)
stage: ball_state
---

# 19 · Ball state from several seeds and views of the temporal network

## Goal
Find out whether averaging the temporal network's stoppage log-odds over training seeds and/or a
home/away-swapped test-time view makes version 4's ball state better on the matches we can score.
A single network is one draw of its initialisation and of the order of its training windows; the
average of a few draws has lower variance, which could steady the stoppage boundaries that spellF1
rewards.

## Context
Version 4 (`configs/c14_best.toml`) takes its ball state from `blend(sequence, kickoff_wait)`;
`sequence` is the dilated TCN of task 13 (`possession/sequence.py`), trained on 7 degraded IDSSE
matches, two noise draws, training seed 0, committed in models/sequence/. Leading teams average
several seeds and views of their temporal networks: Team Sigue three seeds by four views of a
bi-GRU, TercerHombre a multi-seed MS-TCN (research/COMPETITORS.md). The idea is theirs; the
implementation is ours, on our own network.

## Scope
- May touch: `src/possession/sequence_seeds.py` (new), `src/possession/stages/ball_state.py`
  (new method `sequence_ensemble`), `src/possession/cli.py` (`train-sequence --seed`),
  `models/sequence_seeds/` (only for a candidate), `configs/t19_*.toml`, tests, docs.
- Out of scope: `sequence.py`, `sequence_train.py` (their source is in every key), `regime.py`,
  `solution/`, organiser files, models/sequence/ (the committed network and its listing). Version
  4's file must stay byte-identical.

## Interface
- `ball_state.sequence_ensemble(seeds=[0, 1, 2], tta=false)`: the mean stoppage log-odds over the
  networks of `seeds` (and each one's swapped view with `tta`); `seeds=[0]`, no `tta` = `sequence`.
- `possession train-sequence --seed N [--seed M] --folds --final` (needs torch): the network with
  training seed N, out-of-fold IDSSE predictions under `.cache/sequence/<key>/` (and their swapped
  views), final network in models/sequence_seeds/<key>.npz with its own sha256 listing.
- Family (ball state `blend([sequence_ensemble(...), kickoff_wait])`, everything else version 4's):
  `t19_s1_tta` (seed 0 + swapped view), `t19_s2` (seeds 0, 1), `t19_s2_tta`, `t19_s3` (seeds 0, 1,
  2), `t19_s3_tta`; plus `t19_s1` (seeds [0], no view), which must equal version 4 cell for cell.

## Decision rule, written 2026-11-15 before any number of this family was measured
A change from version 4 is adopted only if all of the following hold:
1. Rule D (tasks/15-match-matched-evaluation.md) over the family:
   `uv run python scripts/grid14.py decide-d t19_s1_tta t19_s2 t19_s2_tta t19_s3 t19_s3_tta
   --reference c14_best --bins truth:c14_best`. Evidence: the four Metrica profiles (label_free,
   match_like, match_regime, match_regime_still), readings 2/3 `controlled`, 1/3 `touch`; a member
   is chosen on two Metrica matches and scored on the third.
2. Held-out weighted gain over version 4 at least max(0.005, 0.003 x sqrt(2 ln K)) with K = 6, every
   member including version 4 counted (0.0057; the script prints the value for K = 5, 0.0054, and
   the stricter 0.0057 applies), and the 5% bound of the weighted block bootstrap above 0.
3. Floors: no (profile, reading) below -0.010; PFF weighted at least -0.005; plain IDSSE and
   `idsse_auto` weighted at least -0.010.
4. Footprint on the match inside: 3.5 to 7 possessions per live minute, dead share 0.35 to 0.50.
5. The same verdict with every window weighing 1 (`--bins unit`).
Reported, not deciding: the chosen member against version 5 (`d15_s28_l6_t6`) with `compare
--weighted`. Otherwise version 4 stays. `t19_s1` must reproduce version 4's cells exactly (a check
of the plumbing, not a member).

## Acceptance criteria
- [ ] `sequence_seeds.py` + `sequence_ensemble`, focused tests, `make check` green
- [ ] seed 0's key is the committed network's; V4 byte-identical; seeds 1 and 2 trained (folds + final)
- [ ] the family measured on the 7 sets; rule applied; verdict logged
- [ ] runs logged for the chosen member; docs/DECISIONS.md lines (with credit)
- [ ] networks committed only for a candidate, with the exact retrain command

## Log
- 2026-11-15 · Claude · Task opened, decision rule written before measuring. Seed 0's key
  (`sequence_seeds.run_key(0)`) is a2d3076d6d2c4fd1, the committed network's; seeds 1 and 2 get
  504c3a8b8473a074 and 9eb7d308605d6ec7. Training of seeds 1 and 2 started (one slot, sequential).
