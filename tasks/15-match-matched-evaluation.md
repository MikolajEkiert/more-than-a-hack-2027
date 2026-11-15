---
status: doing
owner: MikolajEkiert (Claude)
stage: infra, decode
---

# 15 · An evaluation matched to the competition match: reproducibility on Linux, composition-weighted scores, a stress test for its ball noise, and version 4's stoppage settings

## Goal
Make the internal evaluation estimate what the official score measures on this match: the file must
regenerate on another machine, validation windows count as much as they resemble the match's
possession structure, the match's own kind of ball noise is tested, and version 4's stoppage settings
are re-checked under that evaluation.

## Context
The official score is spellF1 of one match (BAR-ATH). Read through our own labels (aggregates only),
the match is stoppage-rich: 12.1-12.8 stoppages per 10 graded minutes against 9.6-10.7 in every
validation truth; 45-47% of its possessions start right after a stoppage (validation truths 0.28-0.46);
its stoppages are shorter (median 12-16 s against 15.5-20 s); 20% of its 10-minute blocks have 15 or
more stoppages, a density no Metrica window and almost no PFF or IDSSE window reaches. Its away
possessions are short (median 5.6-6.4 s against 9.4-9.9 s at home). Its tracked ball lies still twice
as often as in the profile fitted to its ball regime (15.5 against 7.1 runs per 10 minutes), which no
degradation simulates. Version 4's gain over version 1 grows with a window's stoppage density
(`label_free`: +0.040 / +0.047 / +0.068 for fewer than 9 / 9-12 / 12-15 stoppages per 10 minutes).
Version 4's file (sha256 6fd87cc4...) was regenerated on macOS arm64 only. The owner fixed the
weights of the two readings for this task: 2/3 `controlled`, 1/3 `touch`.

## Scope
- May touch: `scripts/repro_container.sh` (new), `src/possession/composition.py` (new),
  `scripts/composition15.py` (new), `scripts/grid14.py` (weighted comparison, `decide-d`),
  `src/possession/validation.py` (an optional still-ball mechanism, off by default),
  `configs/degradation/match_regime_still.json` (new), decoder-variant configs, tests, docs.
- Out of scope: `solution/`, `regime.py`, `sequence.py`, `sequence_train.py`, organiser files,
  the existing degradation profiles (they must stay byte-identical), any retraining.

## Rule D, written before any number of steps 3-4 was measured (2026-11-15)
For a change to version 4 (`configs/c14_best.toml`):
1. Evidence: the three Metrica profiles (`label_free`, `match_like`, `match_regime`) and, once it
   exists, `match_regime_still`; readings weighted 2/3 `controlled`, 1/3 `touch`; every 10-minute
   window weighted by the share of the match's blocks in its stoppage-density bin (fewer than 9, 9-12,
   12-15, 15 or more stoppages per 10 minutes) over that bin's share among the set's windows. A member
   of a family is chosen on two Metrica matches and scored on the third.
2. Accept if the held-out weighted gain over version 4 is at least max(0.005, 0.003 x sqrt(2 ln K)),
   K the members compared, and the 5% bound of a weighted bootstrap over (match, window) blocks is
   above 0.
3. Floors instead of win counts: no (profile, reading) below -0.010; PFF at least -0.005; plain IDSSE
   and `idsse_auto` at least -0.010.
4. Footprint on the match: 3.5 to 7 possessions per live minute, dead share 0.35 to 0.50.
5. Otherwise version 4 stays.
Stress test (step 3): version 4 must keep its advantage over version 1 under `match_regime_still`
(difference at least 0 with the 5% bound above -0.010); if it does not, the component that breaks is
found (M6, `d_team_m6`) and only the robust part is kept, decided by this rule.

## Acceptance criteria
- [ ] version 4's file regenerated in Linux arm64 and x86-64 containers, same sha256
- [ ] composition-weighted comparison (`grid14.py compare --weighted`, `decide-d`), tested; the
      unweighted path unchanged
- [ ] `match_regime_still` calibrated without labels; version 1, M6, d_team_m6, version 4 measured
- [ ] version 4's stoppage settings re-checked by rule D
- [ ] `make check` green; docs/DECISIONS.md, docs/FUSION.md section 13, TECHNICAL_REPORT updated

## Log
- 2026-11-15 · Claude · task opened from the approved plan; rule D written before measuring.
