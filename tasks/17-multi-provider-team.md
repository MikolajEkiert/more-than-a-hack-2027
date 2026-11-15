---
status: doing
owner: MikolajEkiert (Claude)
stage: team
---

# 17 · Train the team model on more providers: IDSSE + PFF, and SkillCorner if its labels fit

## Goal
Find out whether version 4's team evidence (LightGBM, features `v1+team`, trained on the seven
IDSSE matches only) gets better when it is also trained on other providers' matches. A model that
has seen several kinds of tracking noise and several teams' styles could read the team in
possession more reliably on a match from a provider it never saw, and a wrong team spell costs two
possessions in spellF1.

## Context
- Version 4 (`configs/c14_best.toml`): team evidence from `team.supervised(features="v1+team")`,
  trained by `supervised._train_final_team` on the seven IDSSE matches (two noise draws, both
  orientations, every 5th graded frame, the package's sample weights). Committed model
  `models/team/57471b9e1a31bec96010.txt.gz`; it must keep being used, byte for byte.
- Training the package's own model (features `v1`) on the 48 PFF training matches as well, or on
  them alone, was measured in task 10 and dropped. It was never tried with the `v1+team` features,
  which are what version 4 uses.
- The idea of training on several providers comes from other teams (research/COMPETITORS.md
  section 1: Barca-Hackathon, Team Sigue, SAM, Anfield train on DFL plus SkillCorner open data
  and/or Metrica). Metrica is our validation set and is never trained on.
- Validation: Metrica `label_free`, `match_like`, `match_regime`, `match_regime_still`; PFF 16 fixed
  test matches (`supervised.pff_games()[1]`, never trained on); IDSSE out of fold, plain and
  `idsse_auto`. Cells of version 4 (`c14_best`), version 1 and version 5 (`d15_s28_l6_t6`) are in
  `.cache/family/`.

## Scope
- May touch: `src/possession/supervised.py` (a new training-data option for the team-feature
  model, off by default), `src/possession/skillcorner.py` (new, only if step B passes its gate),
  `configs/t17_*.toml`, tests, `docs/DECISIONS.md`, `docs/FUSION.md`, this file.
- Out of scope: `solution/`, `evaluation/`, `starter/`, `regime.py`, `sequence.py`,
  `sequence_train.py`; version 4's path (its cache key, its committed model and its file must stay
  byte-identical); Metrica as training data; the 16 PFF test matches as training data.

## Interface
- `team.supervised(features="v1+team", data=...)`: `data` names the training matches of the
  team-feature model. "idsse" (default) is version 4's, unchanged. "idsse+pff" adds the 48 PFF
  training matches in the package's proportions for extra data (one noise draw, one orientation,
  every 15th graded frame, the same degradation topped up for the ball PFF already lacks).
  Step B would add "idsse+sc" and "idsse+pff+sc".
- IDSSE out of fold: each fold trains on the other six IDSSE matches plus all the extra matches;
  the held-out IDSSE match is never in.
- Configs: `configs/t17_pff48.toml` (version 4 with that team model), and in step B
  `t17_sc.toml`, `t17_pff48_sc.toml`.

## Decision rule, written before measuring (2026-11-15)
Reference: version 4 (`c14_best`). Members: every `t17_*` variant measured in this task
(step A: `t17_pff48`; step B adds `t17_sc` and `t17_pff48_sc` only if SkillCorner passes the
label-compatibility gate below, which is judged on label aggregates before any variant is scored).
K = the number of members.

1. `uv run python scripts/grid14.py decide-d <all members> --reference c14_best --bins truth:c14_best`
   (rule D of task 15): evidence the four Metrica profiles, readings 2/3 `controlled`, 1/3 `touch`,
   windows weighted by the match's stoppage-density composition (match binned by version 4's
   labels); a member is chosen on two Metrica matches and scored on the third.
2. Accept if the held-out weighted gain is at least max(0.005, 0.003 x sqrt(2 ln K)) and the 5% bound
   is above 0.
3. Floors: PFF at least -0.005, plain IDSSE and `idsse_auto` at least -0.010, no (profile, reading)
   below -0.010.
4. Footprint on the match inside 3.5-7 possessions per live minute and dead share 0.35-0.50.
5. Otherwise version 4 stays.
Also reported, not deciding: the same with `--bins unit`, and every member against version 5.
Caveat fixed in advance: a member trained on PFF matches is scored on PFF test matches of the same
tournament and provider, so its PFF number is in-domain and is used only as a floor, never as
evidence of a gain.

SkillCorner gate (step B, before any training on it): its per-frame possession information must
give H / A / D labels whose aggregates resemble those of our training and validation labels:
possessions per live minute within the range of the IDSSE and Metrica readings we score (about 4
to 8.3), and a dead share of 0.25 to 0.50 of graded time, with D derivable from the data itself
(not imputed wholesale). If not, step B stops and the reason is recorded.

## Acceptance criteria
- [ ] `data="idsse+pff"` for the team-feature model; version 4's model key, model and file unchanged
- [ ] focused tests; `make check` green
- [ ] `t17_pff48` measured on all 7 sets; rule applied; run logged with a NOTE
- [ ] SkillCorner compatibility reported (aggregates); step B done or stopped with the reason
- [ ] docs/DECISIONS.md lines (credit to the other teams for the idea)

## Log
- 2026-11-15 · Claude · task opened; decision rule written before measuring.
