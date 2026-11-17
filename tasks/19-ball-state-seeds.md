---
status: done
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
- [x] `sequence_seeds.py` + `sequence_ensemble`, focused tests, `make check` green
- [x] seed 0's key is the committed network's; V4 byte-identical; seeds 1 and 2 trained (folds + final)
- [x] the family measured on the 7 sets; rule applied; verdict logged
- [x] runs logged for the chosen member; docs/DECISIONS.md lines (with credit)
- [x] networks committed only for a candidate (none is), with the exact retrain command

## Log
- 2026-11-15 · Claude · Task opened, decision rule written before measuring. Seed 0's key
  (`sequence_seeds.run_key(0)`) is a2d3076d6d2c4fd1, the committed network's; seeds 1 and 2 get
  504c3a8b8473a074 and 9eb7d308605d6ec7. Training of seeds 1 and 2 started (one slot, sequential).
- 2026-11-15 · Claude · Swapped view without training: `sequence_seeds.swap_folds(0)` runs each
  seed-0 fold network on the swapped features of its held-out IDSSE copy (after checking that it
  reproduces the stored fold prediction bit for bit); PFF and Metrica use the package's swapped
  feature cache, as the LightGBM's `tta`. Training of seeds 1 and 2 (folds, swapped folds, final),
  one slot, 12:25 to 15:15: about 85 minutes per seed. `t19_s1` (seeds [0], no view) reproduces
  version 4's cells exactly on label_free and PFF.
- 2026-11-15 · Claude · Differences against version 4 (unweighted mean [5%, 95%] block bootstrap,
  matches won; weighted by `truth:c14_best` in the DECISIONS line where it matters):

  | set | s1_tta | s2 | s2_tta | s3 | s3_tta |
  |---|---|---|---|---|---|
  | label_free | +0.0033 [+0.001, +0.008] 2/3 | +0.0020 [-0.002, +0.007] 2/3 | +0.0054 [+0.002, +0.011] 3/3 | +0.0030 [-0.002, +0.008] 2/3 | +0.0049 [+0.001, +0.010] 3/3 |
  | match_like | +0.0015 [-0.002, +0.004] 3/3 | -0.0034 [-0.008, +0.001] 1/3 | -0.0003 [-0.004, +0.003] 2/3 | -0.0013 [-0.005, +0.002] 2/3 | +0.0005 [-0.003, +0.004] 2/3 |
  | match_regime | +0.0014 [-0.001, +0.003] 2/3 | +0.0018 [-0.000, +0.004] 3/3 | +0.0021 [-0.000, +0.005] 2/3 | +0.0021 [+0.000, +0.005] 3/3 | +0.0037 [+0.002, +0.006] 3/3 |
  | match_regime_still | +0.0018 [-0.001, +0.005] 3/3 | +0.0001 [-0.004, +0.004] 2/3 | +0.0007 [-0.003, +0.004] 3/3 | -0.0016 [-0.005, +0.002] 0/3 | -0.0014 [-0.005, +0.002] 1/3 |
  | IDSSE | +0.0022 [-0.000, +0.004] 6/7 | -0.0001 [-0.002, +0.002] 4/7 | +0.0019 [-0.001, +0.004] 4/7 | +0.0023 [-0.001, +0.005] 4/7 | +0.0034 [+0.001, +0.006] 5/7 |
  | IDSSE auto | +0.0016 [+0.000, +0.005] 6/7 | +0.0015 [-0.000, +0.004] 6/7 | +0.0026 [+0.001, +0.006] 7/7 | +0.0021 [+0.001, +0.005] 5/7 | +0.0042 [+0.003, +0.008] 6/7 |
  | PFF | +0.0033 [+0.001, +0.006] 11/16 | +0.0009 [-0.002, +0.003] 9/16 | +0.0025 [-0.001, +0.005] 12/16 | -0.0002 [-0.003, +0.003] 6/16 | +0.0003 [-0.004, +0.003] 10/16 |

- 2026-11-15 · Claude · Rule D (`decide-d ... --bins truth:c14_best`): held-out picks t19_s2_tta,
  t19_s1_tta, t19_s2_tta; held-out weighted gain +0.0011 (threshold +0.0057 with K = 6; the script
  prints +0.0054), 5% bound -0.0010; every (profile, reading) above -0.010; floors pass (PFF
  +0.0034, IDSSE +0.0025, auto +0.0042); footprint inside (4.84 per live minute, dead 0.479). With
  unit weights: gain +0.0016, bound -0.0003, chosen t19_s1_tta. **Verdict: version 4 stays.** The
  swapped view is a small, consistent gain (positive on all 7 sets alone), far below the threshold;
  extra seeds add nothing reliable. Against version 5 (`d15_s28_l6_t6`), reported only: t19_s2_tta
  -0.004 to -0.013 on Metrica and IDSSE, +0.007 on PFF (13 of 16), the same pattern as version 4.
- 2026-11-15 · Claude · Run logged for the member chosen on all three matches:
  experiments/runs/20261115T155055_mikolajekiert_t19_s2_tta.json (267 possessions; footprint inside).
  The seed-1 final network retrained from scratch into a scratch folder is byte-identical to the
  first training (same sha256; numpy against torch 7e-15). Networks of seeds 1 and 2 are not
  committed (no candidate); rebuild: `uv run --with torch==2.15.0 possession train-sequence --seed 1
  --seed 2 --folds --final` (writes models/sequence_seeds/ and the out-of-fold files).
