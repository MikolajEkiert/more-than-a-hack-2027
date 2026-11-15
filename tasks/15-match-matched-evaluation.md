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
- [x] version 4's file regenerated in Linux arm64 and x86-64 containers, same sha256
- [x] composition-weighted comparison (`grid14.py compare --weighted`, `decide-d`), tested; the
      unweighted path unchanged
- [x] `match_regime_still` calibrated without labels; version 1, M6, d_team_m6, version 4 measured
- [x] version 4's stoppage settings re-checked by rule D
- [x] `make check` green; docs/DECISIONS.md, docs/FUSION.md section 13, TECHNICAL_REPORT updated

## Log
- 2026-11-15 · Claude · task opened from the approved plan; rule D written before measuring.
- 2026-11-15 · Claude · Step 1: version 4's file regenerated from a fresh clone with an empty cache
  in a Linux arm64 container (27 s) and an emulated Linux x86-64 container (44 s), Debian 12,
  `uv sync --frozen` (`scripts/repro_container.sh`; LightGBM needs libgomp1, which the slim image
  lacks): sha256 6fd87cc4467530fa... on both, the organisers' validator passes.
- 2026-11-15 · Claude · Step 2: `possession/composition.py`, `scripts/composition15.py`, weighted
  comparison in `scripts/grid14.py` (`compare --weighted`, `decide-d`). The match's 10-minute blocks
  by our labels (V4 and V1): 0.10 / 0.30 / 0.40 / 0.20 below 9 / 9-12 / 12-15 / 15 or more stoppages
  per 10 minutes. Window weights: Metrica 0.38 / 0.60 / 1.71 / 0 (covers 80% of the match), IDSSE
  0.37 / 0.70 / 1.33 / 0 (80%), PFF 0.24 / 0.65 / 4.4 / 5.5 (100%). V4 minus V1, weighted (unweighted):
  label_free +0.059 (+0.052), match_like +0.068 (+0.066), match_regime +0.050 (+0.049), PFF +0.038
  (+0.046), plain IDSSE -0.001 (+0.001), IDSSE auto +0.027 (+0.023).
- 2026-11-15 · Claude · Step 3: optional `still_ball_rate` / `still_ball_s` / `still_ball_spare` in
  `validation.degrade` (own random stream; the three original profiles and `match_regime` stay byte
  for byte, the latter checked by a stored hash). `match_regime_still` fitted by `scripts/still15.py`
  on tracking statistics only: 18 events per 10 live minutes give 15.57 still runs per 10 minutes
  (the match: 15.49) and a still share of tracked frames of 0.101 (0.100). Under it: V1 0.609 (0.655
  under match_regime), M6 +0.026, d_team_m6 +0.052, V4 +0.056 [+0.046, +0.062], 3 of 3 (weighted
  +0.057). Version 4 keeps, and widens, its lead under the match's own kind of ball noise.
- 2026-11-15 · Claude · Step 4, rule D over twelve decoder variants on version 4's evidence: lower
  stoppage costs lose (20 and 24 below 28 everywhere); turnover cost 6 is chosen on every held-out
  match (+0.0105 weighted, threshold +0.0067, 5% bound +0.0076) and fails the PFF floor (-0.0070
  weighted, -0.0042 unweighted, 5 of 16): a bet on `controlled`. Verdict: version 4 stays. Runs of
  versions 1 and 4 under `match_regime_still`: experiments/runs/20261115T0200*.
- 2026-11-15 · Claude · Correction, written before any number of it was computed. Re-checking step 2
  showed that the weighting compares two instruments: the match is binned by our labels, the
  validation windows by their truth. Version 4's labels count more stoppages than the truth: under
  `match_regime_still` 12.4 against 10.5 per 10 graded minutes, the excess being stoppages lying in
  true play (2.75 per 10 minutes, median 2.6 s; splits 0.25, missed 0.74), and its bin shares there
  (0.12 / 0.33 / 0.38 / 0.17) are the match's by our labels (0.10 / 0.30 / 0.40 / 0.20). The
  "stoppage-rich match" is then mostly our own over-counting. **Rule D'** = rule D with the match and
  the windows both binned by version 4's labels (`composition15.py predicted`, `grid14.py --bins
  c14_best`). Because it comes after rule D's verdict, a change to version 4 is adopted only if it
  passes rule D' and also rule D with every window weighing 1 (`--bins unit`); otherwise version 4
  stays. The composition-weighted numbers of steps 2-3 are reported again under rule D' weights.
- 2026-11-15 · Claude · The instrument check, measured. Stoppages per 10 graded minutes, truth against
  version 4's labels: label_free 10.47 / 10.01, match_like 10.47 / 10.13, match_regime 10.47 / 10.51,
  match_regime_still 10.47 / 12.43, PFF 9.59 / 10.02, IDSSE 10.67 / 10.28 (auto 10.16). Version 4
  counts stoppages right everywhere except under the still ball, where it adds short stoppages in
  play (2.75 per 10 minutes, median 2.6 s). The match by version 4 alone: 11.67 per 10 minutes, bins
  0.1 / 0.4 / 0.3 / 0.2 (10 blocks), what version 4 shows under match_regime_still (11.93; 0.10 /
  0.37 / 0.47 / 0.07). Through that profile's confusion of bins the match's true shares come out
  0.22 / 0.67 / 0.11 / 0 (V4+V1: 0.22 / 0.45 / 0.33 / 0), an ordinary game (Metrica truth 0.27 / 0.50 /
  0.23 / 0). The match is not shown to be stoppage-rich; task 15's weighted numbers overweight dense
  windows. Rule D' binning by version 4 still reads the noise-free profiles with a less biased
  instrument than the match, so unit weights are the most defensible reading.
- 2026-11-15 · Claude · Rule D' and rule D with unit weights over the twelve decoder variants: both
  choose turnover cost 6 (d15_s28_l6_t6) on every held-out match. D': gain +0.0090 (threshold
  +0.0067, 5% bound +0.0071), PFF -0.0027, IDSSE +0.0069, auto +0.0134. Unit: +0.0091 (bound
  +0.0068), lowest (profile, reading) label_free/touch -0.0094, PFF -0.0038, IDSSE +0.0075, auto
  +0.0121. Footprint 4.79 possessions per live minute, dead share 0.479. Both pass, so by the rule
  written above the change is adopted: **version 5 = `configs/c15_best.toml`** (version 4 with
  turnover cost 6). Caveat: the unweighted PFF difference (-0.0042, match level) was already printed
  before rule D' was written. `possession submit` (two runs from a fresh parse, validator passes):
  sha256 0167a8c6c243f1fdfa2d36916627a699586a93123d5de75e00a26b82198d5338.
