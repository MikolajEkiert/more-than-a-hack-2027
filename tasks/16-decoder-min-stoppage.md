---
status: doing
owner: MikolajEkiert (Claude)
stage: decode
---

# 16 · A minimum stoppage length in the decoder: short dead spells inside one team's possession

## Goal
Remove short stoppages (D spells) that lie between two spells of the same team, which version 4
creates under still-ball noise and which split one true possession into two false ones; spellF1
matches whole possessions, so each such split costs a matched possession.

## Context
Version 4 = `configs/c14_best.toml` (recommended), version 5 = `configs/c15_best.toml` (version 4
with turnover cost 6, the documented alternative). Task 15 (docs/FUSION.md section 13): under
`match_regime_still` version 4 adds 2.75 stoppages per 10 minutes in true play (median 2.6 s), an
excess below 4 s; on the competition match version 4's stoppages show an excess at 4-8 s whose share
with the same team before and after is 0.58 against about 0.47-0.50 in validation. A minimum stoppage
duration in the decoder is used by other teams (Barca-Hackathon, SAM, ZesterJ): the idea is theirs,
credited in docs/DECISIONS.md; the implementation and the measurement are ours.

Duplication check: `switch_cost` already has `short_spells` (short *team* spells between two spells
of the other team) and the snaps of stoppage edges; nothing removes short *dead* spells. No other
task in `status: doing` touches the decoder's post-processing.

## Scope
- May touch: `src/possession/stages/decode.py` (new options of `switch_cost`, off by default),
  `tests/test_decode_options.py`, `configs/t16_*.toml`, docs (DECISIONS, FUSION section 14), this file,
  `experiments/runs/` for a logged run.
- Out of scope: `solution/`, `regime.py`, `sequence.py`, `sequence_train.py`, organiser files,
  `submission/`, any retraining, the ball_state and team stages.

## Interface
`decode.switch_cost(..., min_stop_s=0.0, min_stop_min_alive=0.0)`: a D spell shorter than
`min_stop_s` whose neighbours in the same half are spells of one and the same team becomes that team;
with `min_stop_min_alive` above zero only if the mean `p_alive` over the spell is at least that value
(the decoder's own dead-ball evidence is weak there). A spell between two different teams, at a half
edge or next to X is never touched. Applied after the snaps, before `short_spells`.

Family (16 members, written before measuring): `min_stop_s` in {2, 3, 4, 6} x guard in {none,
`min_stop_min_alive` 0.2} x decoder of version 4 (turnover 5) or version 5 (turnover 6); names
`t16_t{5,6}_m{2,3,4,6}[_a20]`. Why 0.2: between two spells of one team the Viterbi path pays stop plus
restart cost (28 nats) for a stoppage, so a 2-4 s stoppage needs a mean p_alive of roughly 0.05-0.15
over it; a spell with a mean above 0.2 owes its existence mostly to the edge snaps or to absent team
evidence, the weakest kind.

## Decision rule, written 2026-11-15 before any member was measured
For a change to version 4:
1. `uv run python scripts/grid14.py decide-d <16 members> --reference c14_best --bins truth:c14_best`
   (task 15's rule D with its final instrument: windows binned by their truth, the match by version
   4's labels; readings 2/3 `controlled`, 1/3 `touch`; evidence label_free, match_like, match_regime,
   match_regime_still; the member chosen on two Metrica matches and scored on the third).
2. Accept if the held-out weighted gain is at least max(0.005, 0.003 x sqrt(2 ln K)), K = 16 members
   (threshold +0.0071), and its 5% bootstrap bound is above 0.
3. Floors: PFF at least -0.005; plain IDSSE and `idsse_auto` at least -0.010; no (profile, reading)
   below -0.010 (all weighted, as `decide-d` prints them).
4. Footprint on the match inside 3.5-7 possessions per live minute, dead share 0.35-0.50.
5. Otherwise version 4 stays. Reported but not deciding: the same rule with `--bins unit`, and every
   member against version 5 (`grid14.py compare ... --against d15_s28_l6_t6`, version 5's cells).
   A member on version 5's decoder that passes inherits version 5's PFF caveat; it is reported as
   such. The public 5-minute window plays no part.

## Acceptance criteria
- [ ] options in `switch_cost`, off by default; version 4 byte-identical (contract tests, a focused
      test, and a config equal to c14_best reproducing its cells through the fast path)
- [ ] 16 members measured on the 7 sets
- [ ] stoppage-length diagnostic before / after (`composition15.py lengths`)
- [ ] rule applied; run logged for a chosen member if any
- [ ] `make check` green; docs/DECISIONS.md line (credit to the other teams), FUSION.md section 14

## Log
- 2026-11-15 · Claude · task opened; decision rule and family written before measuring.
