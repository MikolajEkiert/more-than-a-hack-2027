---
status: doing
owner: MikolajEkiert (Claude, with sub-agents)
stage: ball_state, team, decode, infra
---

# 14 · New directions after tasks 12-13: fusion of the temporal network's team head, decoder options, a validation profile in the match's ball regime, team features

## Goal
Find the configuration with the best expected spellF1 on the competition match, using only ideas
not taken from other participants, and check whether the way versions are verified fits the match.

## Context
Version 1 (`configs/hybrid_supervised.toml`) is submitted (sha256 `25f146c0...`). Everything
measured after it failed the rule of docs/FUSION.md section 10. The biggest loss is the team in
control of a tracked ball (oracle +0.225 under the harsh profile); then stoppages (+0.07) and the
team without a ball (+0.08). Findings of the preparation (label-free, `possession inspect regime`
with the hybrid's labels): on the match the ball is missing on about 8% of live frames and 75% of
dead ones; the evidence profiles lose it on 16% (`label_free`) and 26% (`match_like`) of live
frames, plain IDSSE on 41% at random, PFF on 6%. The temporal network's team head
(`team.sequence`) has never been measured.

## Scope
- May touch: new configs, `scripts/grid14.py`, `src/possession/stages/decode.py` (new parameters
  whose defaults keep every current output byte-identical), `src/possession/validation.py`
  (new optional degradation mechanisms; existing profiles must give byte-identical degraded
  matches and identical profile dicts), new profiles in `configs/degradation/`,
  `src/possession/supervised.py` and new modules for an extra feature block, tests, docs.
- Out of scope: `solution/` (merged as written, hashed into its caches), `regime.py`,
  `sequence.py`, `sequence_train.py` (their source is part of cached model keys), organiser files.

## Decision rules, written before any number of this task was measured (2026-11-14)
**Rule A, the team's rule (docs/FUSION.md section 10), unchanged.** Every candidate is reported
under it.

**Rule B, a rule matched to the match's ball regime**, proposed in this task and reported beside
rule A; the owner chooses which one decides. Disclosure: it was written after seeing the results
of tasks 12-13 (M6), so it cannot be blind to them.
1. A candidate is a whole configuration, compared with version 1 by paired differences.
2. Sets: Metrica `label_free`, `match_like` and (if it exists in time) a profile fitted to the
   match's ball regime, both readings at 50/50 (evidence); PFF test matches (real noise); IDSSE in
   the automatic ball regime (`idsse_auto`); plain IDSSE (random ball loss, the least like the
   match).
3. Where a candidate was picked from several, it is scored held out (chosen on two Metrica matches,
   scored on the third), as in rule A.
4. Accept if: the mean held-out difference over the Metrica evidence is at least +0.010 with the
   5% bound of a paired bootstrap over (match, 10-minute window) blocks above 0; no (profile,
   reading) below -0.010; PFF mean >= 0 with at least 9 of 16 matches won; `idsse_auto` >= 0;
   plain IDSSE >= -0.010; footprint on the match inside 3.5-7 possessions per live minute and a
   dead share of 0.35-0.50.

## Interface
- `uv run python scripts/grid14.py measure|compare|decide|footprint ...` (cells shared with
  `scripts/family_grid.py` in `.cache/family/`).

## Acceptance criteria
- [ ] every direction measured on all sets, numbers in the log below and in docs/DECISIONS.md
- [ ] the decision under rule A and rule B for the best candidate
- [ ] an assessment of the verification method itself
- [ ] `make check` green; the submitted file unchanged unless the owner decides otherwise

## Log
- 2026-11-14 · Claude · task opened; rules A and B written before measuring.
- 2026-11-15 · Claude (team features) · **Direction: features for the team in control of a visible,
  noisy ball.** `possession/team_features.py` computes 73 columns from the package's container
  (each half oriented as the package does), appended to its 203 (feature set `v1+team` of
  `supervised.py`; parameter `features` of `ball_state.supervised` / `team.supervised`, threaded
  through `probabilities`, `model`, `_predict` and both held-out paths; "v1" is untouched, the
  submitted file keeps sha256 `25f146c0`). Per team: players goal-side of the ball (all, and
  within 20 m across; counts, shares, difference), the same against the last / next tracked ball
  where it is missing; |v_ball - v_player| of its nearest player and whether the same track was
  nearest 0.4 s earlier (means over +-0.4 s); the smallest per-player mean distance to the ball
  over +-0.2 / 0.6 / 1.2 s; mean speed toward the ball within 15 m and players closing in faster
  than 1.5 m/s within 10 m; spread (sd of x, of y, product) against the team's own median / IQR in
  the half; past 1 / 4 s, future 1 / 4 s and centred 1 s means of seven of these. Parameters fixed
  before training; one change made on training data only (the same-track cue left out of the
  context means: its class means hardly differ on IDSSE). Everything else as version 1's model:
  calibrated degradation, raw labels, noise seeds 0 and 1, both orientations, every 5th graded
  frame, sample weights, `LGB_PARAMS`, 300 rounds; 14 fold models (per seed, six matches predict
  the seventh, in both input regimes) and the final model, cached in `.cache/team_features/`. The
  training rows were checked bit for bit against the package's `ev.stack` on the real prepared
  matches (`scripts/team_block.py check`). Timings: the block 2 s per IDSSE match; final model
  269 s (LightGBM's default 8 threads, before the machine's cap); 13 of the 14 fold models
  1290 s at `OMP_NUM_THREADS=4` (fold 0 of seed 0 at 8 threads). Gain share of the block in the
  final model 6.95% (all 73 used; best block column 19th; shape against the team's norm leads).
- 2026-11-15 · Claude (team features) · Members, spellF1 minus version 1 (5-95% interval over
  (match, 10-minute window) blocks; readings; matches won):

  | set | d_team (team from `v1+team`) | d_both (both stages from `v1+team`) | d_team_m6 (ball state of M6) |
  |---|---|---|---|
  | `label_free` | +0.022 [+0.015, +0.028]; touch +0.016, contr. +0.028; 3/3 | +0.022 [+0.014, +0.028]; +0.017, +0.027; 3/3 | +0.049 [+0.037, +0.055]; +0.041, +0.056; 3/3 |
  | `match_like` | +0.037 [+0.031, +0.043]; +0.033, +0.042; 3/3 | +0.035 [+0.027, +0.040]; +0.031, +0.038; 3/3 | +0.059 [+0.047, +0.066]; +0.056, +0.061; 3/3 |
  | IDSSE out of fold | +0.001 [-0.006, +0.004]; c1.5 +0.003, c3.5 -0.002; 2/7 | +0.001 [-0.006, +0.005]; +0.003, -0.002; 3/7 | +0.000 [-0.007, +0.005]; +0.002, -0.002; 3/7 |
  | `idsse_auto` | -0.002 [-0.007, +0.002]; -0.001, -0.002; 3/7 | +0.001 [-0.004, +0.006]; +0.002, -0.000; 4/7 | +0.023 [+0.017, +0.031]; +0.025, +0.021; 7/7 |
  | PFF test matches | +0.001 [-0.003, +0.005]; 9/16 | +0.009 [+0.005, +0.014]; 12/16 | +0.046 [+0.039, +0.055]; 16/16 |

  d_team_m6 minus M6 (the block alone, on M6's ball state): `label_free` +0.024 (3/3), `match_like`
  +0.041 (3/3), IDSSE +0.001 (4/7), `idsse_auto` -0.003 (1/7), PFF +0.002 (10/16). The mechanism
  is the intended one: team accuracy on frames both call live 0.923 -> 0.932 (`label_free`),
  0.877 -> 0.894 (`match_like`), 0.972 -> 0.973 (PFF); predicted possessions 324 -> 310 under
  `match_like` (truth 298.5). The block gains where the ball is off its possessor (Metrica
  offsets 0.5 and 2 m) and is neutral where the ball is as precise as in training (IDSSE, both
  input regimes) or the team is already easy (PFF). Footprint on the match (V1: 300 possessions,
  5.59 per live minute, dead 0.4935): d_team 286, 5.31, 0.4918 (2.1% of frames differ from V1);
  d_both 283, 5.23, 0.4887 (2.8%); d_team_m6 276, 5.01, 0.4796 (7.7%); all inside.
- 2026-11-15 · Claude (team features) · **Verdicts.** Rule A on {d_team, d_both, d_team_m6}: the
  nested choice picks d_team_m6 on every held-out match (held-out gain +0.049 / +0.059, every
  reading positive, 3 of 3), PFF passes, footprint inside, **IDSSE veto fails** (+0.000, 3 of 7
  won; 5 needed): version 1 stays. Rule B on the same family: held-out evidence +0.054 (5% bound
  +0.045), PFF +0.046 (16/16), `idsse_auto` +0.023, IDSSE +0.000 (>= -0.010), footprint inside:
  **change to d_team_m6**. Taken alone: d_team fails A (IDSSE 2 of 7) and B (`idsse_auto`
  -0.002); d_both fails A (IDSSE 3 of 7) and passes B. On the method: rule A's IDSSE veto asks a
  change to *win* 5 of 7 matches on the one set whose ball is as precise as the training data's;
  a change aimed at a noisy ball is neutral there by construction (all three members within
  +-0.001, intervals +-0.006), so the veto measures luck, not harm. Rule B's "no loss beyond
  0.010" on plain IDSSE fits that set's role.
- 2026-11-15 · Claude (team features) · Check for under-fitting, not a member of the decided
  family: d_team with the model trained for 600 rounds (`d_team_r600`, feature set
  `v1+team-r600`; final 387 s, 14 folds 1875 s, at 4 threads; block gain share 7.98%). Minus
  d_team: `label_free` +0.002 [-0.001, +0.006] (2/3), `match_like` +0.009 [+0.002, +0.013] (3/3),
  IDSSE -0.001 (4/7), `idsse_auto` +0.001 (3/7), PFF +0.002 (11/16). Minus version 1: +0.024,
  +0.046, -0.001 (4/7), -0.001 (3/7), +0.003 (8/16); footprint 5.33 possessions per live
  minute, dead 0.4914. Mildly under-fitted under the harsh profile only; alone it fails rule A
  (IDSSE -0.001, PFF 8 of 16) and rule B (PFF 8 of 16, `idsse_auto` -0.001).
