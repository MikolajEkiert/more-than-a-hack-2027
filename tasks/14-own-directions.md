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
