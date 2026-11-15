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
