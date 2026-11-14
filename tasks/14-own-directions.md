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
