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
- [x] an assessment of the verification method itself (docs/FUSION.md section 11)
- [ ] `make check` green; the submitted file unchanged unless the owner decides otherwise

## Log
- 2026-11-14 · Claude · task opened; rules A and B written before measuring.
- 2026-11-15 · Claude · V1, M3, M5, M6 re-measured with window scores (same means as before, so the
  pipeline is deterministic). Paired intervals (5-95%) over (match, window) blocks: M5 is above V1
  on all five sets (label_free +0.023 [+0.015, +0.028], match_like +0.012 [+0.006, +0.015], IDSSE
  +0.004 [+0.002, +0.008] 7/7, IDSSE auto +0.025 7/7, PFF +0.039 16/16); M6 the same except plain
  IDSSE -0.001 (3/7). Footprints: M5 5.31 possessions per live minute, dead 0.484; M6 5.25, 0.479.
- 2026-11-15 · Claude · Minimum-Bayes-risk decoding (`decode.mbr`, src/possession/stages/mbr.py):
  possessions chosen for the largest expected spellF1 under the posterior of the switching-cost
  model, D elsewhere. Screened on Metrica `label_free`, seed 0 (V1 0.662 on the same cells):
  sharpness 0.5 collapses (3-6x too many possessions), 1.0 gives 0.60-0.61, 2.0 0.644, 3.0-5.0
  0.650-0.652. It converges to the Viterbi path as the posterior sharpens and never beats it: the
  posterior of this model is not calibrated well enough to optimise expected spellF1 on. Dropped.
- 2026-11-15 · Claude · The temporal network's team head (never measured before) hurts in every
  form: alone -0.057 (label_free), -0.071 (PFF); blended 0.5/0.5 with the LightGBM -0.010
  (label_free), -0.016 (match_like), -0.022 (IDSSE); 1/1 -0.034 (label_free). Probably its 5 Hz
  output, interpolated to 25 fps, blurs the moment of a change. Dropped. It also trades one reading
  for the other (0.5/0.5 on label_free: touch +0.012, controlled -0.031), i.e. more short spells.
- 2026-11-15 · Claude · Team evidence of the mixed-regime LightGBM (M3's model, never used for the
  team before): alone +0.005 / +0.002 / +0.003 / +0.000 / +0.003 (label_free, match_like, IDSSE,
  IDSSE auto, PFF); averaged with V1's LightGBM 0.5 / 0.5 (`f1_mixteam_v1`) +0.006 [+0.002,
  +0.008] 3/3, +0.002 2/3, +0.002 5/7, +0.001 5/7, +0.004 [+0.002, +0.008] 9/16. Small and
  consistent: an ensemble effect, kept as a bundle candidate.
- 2026-11-15 · Claude · Resource note: three workers plus four measurement processes filled the
  12 GB swap (load average above 200). Measurements now run one process at a time with
  OMP_NUM_THREADS=2; the remaining cells of the team-head members were not measured (dropped).
- 2026-11-15 · Claude · Ball-state family, LightGBM / temporal network weights 0.7/0.3 (`bs_w73`),
  0.5/0.5 (M5), 0.3/0.7 (`bs_w37`), 0/1 (M6), 1/1 (`bs_w11`). More weight on the network is better on
  Metrica and PFF; plain IDSSE prefers some LightGBM. label_free +0.019 / +0.023 / +0.024 / +0.025 /
  +0.022; match_like +0.009 / +0.012 / +0.015 / +0.018 / +0.004; IDSSE +0.003 / +0.004 / +0.002 /
  -0.001 / +0.003; IDSSE auto +0.022 / +0.025 / +0.026 / +0.026 / +0.026; PFF +0.032 / +0.039 /
  +0.042 / +0.044 / +0.036. Nested choice picks M6 on every held-out match. Rule A: version 1
  stays (match_like +0.018 < +0.02, IDSSE veto -0.001, 3/7). Rule B: change to M6 (held-out mean
  +0.021, 5% bound +0.015; PFF 16/16, IDSSE auto 7/7, plain IDSSE -0.001 within -0.010; footprint
  5.25 per live minute, dead 0.479).
- 2026-11-15 · Claude · A label-free referee on the competition match (`scripts/disagree14.py`):
  where two versions disagree on play / stoppage, the share of frames with running players
  (1 s mean of the 90th percentile of player speed >= 2.5 m/s; 0.19-0.23 in true stoppages and
  0.83-0.92 in true play on the labelled sets; on the match 0.137 where V1 and M6 both say dead
  and 0.887 where both say live). V1 vs M6: V1 live / M6 dead on 2.1% of graded frames (ball
  tracked 53%, running 0.65); V1 dead / M6 live on 3.6% (ball tracked 88%, running 0.42). Read
  naively, M6's extra stoppages look like play and its extra play like stoppages, i.e. on the
  match M6 would be worse than V1 where they disagree, mostly on stoppages with the ball in view
  (which the match has on 25% of its dead frames, Metrica on 4-13%, PFF on 1%, plain IDSSE 58%).
  The referee is being calibrated on the labelled sets before any conclusion.
- 2026-11-15 · Claude · Hypothesis from that: V1's LightGBM (trained where the ball is in view
  during 58% of stoppage frames) reads stoppages with a tracked ball better, the network reads
  stretches without a ball better. New method `ball_state.blend_by_ball` (weights by ball
  visibility); members `gate_s55`, `gate_s73`, `gate_s10`, `gate_s73_u37` (tracked-ball weights
  LightGBM/network 0.5/0.5, 0.7/0.3, 1/0, 0.7/0.3; no-ball weights 0/1 except the last, 0.3/0.7).
- 2026-11-15 · Claude · The referee calibrated on the labelled sets (`disagree14.py --set`), V1 vs
  M6, cell "V1 live / M6 dead" (share, ball tracked, running, truly dead) and "V1 dead / M6 live":
  PFF 0.9% / 12% / 0.67 / 0.76 and 2.3% / 84% / 0.26 / 0.10; label_free 2.1% / 26% / 0.77 / 0.65 and
  1.7% / 55% / 0.65 / 0.10; match_like 5.3% / 61% / 0.53 / 0.56 and 0.8% / 10% / 0.63 / 0.13;
  plain IDSSE 1.4% / 11% / 0.74 / 0.46 and 1.1% / 21% / 0.63 / 0.51. Where the versions disagree
  (moments of transition) the running share does not tell play from stoppage: M6 is right in
  56-90% of those frames on PFF and Metrica, half of them on plain IDSSE. The match's "V1 dead /
  M6 live" cell (ball tracked 88%, running 0.42) looks like PFF's (84%, 0.26, 90% live). So the
  naive alarm was wrong; a label-free statistic of the match has to be calibrated on labelled
  sets in the very cells where versions disagree before it can arbitrate between them.
- 2026-11-15 · Claude (verification) · `scripts/regime_compare.py`: the same vector of aggregates
  (`inspect_data.regime_vector`) on the match and on every validation set, through version 1's
  labels, plus the agreement of version 1 with `unsup_v2`; similarity table in docs/FUSION.md
  section 11. No set resembles the match on everything; PFF is closest on the ball lost in play,
  plain IDSSE farthest; the match shows the ball during a quarter of its stoppage time (placed
  before the restart in a third of its stoppages), which no profile did.
- 2026-11-15 · Claude (verification) · `validation.degrade`: optional mechanisms (placed and
  carried ball before restarts, late pick-up after restarts, lead-in share), off by default, on
  a second random stream; the existing profiles are byte-identical on all 27 real conditions.
- 2026-11-15 · Claude (verification) · profile `match_regime` fitted (`regime_compare.py fit
  --write`, 56 settings): the match's ball regime reproduced (P(missing | play) 0.068 against
  0.077, P(missing | stoppage) 0.80 against 0.75, placed ball before 35% of restarts, ball 1.8 m
  from the nearest player against 1.9 m), its hardness not (possession agreement 0.56 against
  0.46). Under it: V1 0.655; M3 +0.010, M5 +0.019, M6 +0.021, each 3 of 3 matches. Rule B, as
  written and with `match_regime` added, chooses M6; rule A keeps version 1.
- 2026-11-15 · Claude (verification) · assessment of the protocol (`scripts/protocol_check.py`):
  standard error of a profile's mean about 0.004; rule A's +0.02 is five of them (a true +0.02
  passes 27% of the time); a member's effect varies between sets three to four times more than
  that and follows the set's ball loss in play; plain IDSSE orders ball-state members unlike every
  other set; about 130 configurations on the same three Metrica matches. Rule C proposed (docs/
  FUSION.md section 11), post hoc: it would choose M6. The owner decides.
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
