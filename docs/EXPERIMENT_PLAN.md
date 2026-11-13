# Experiment plan

Status: draft, 2026-11-13. Deadline: 2026-11-17. Written from the organiser documents
(CHALLENGE, EVALUATION, RULES, FAQ, DATA, SUBMISSION) and a read of publicly visible forks.
No competition-match data was used or seen to write it.

## 1. What the task really is

- One label per video frame (`H` / `A` / `D` / `X`), 199,319 rows; 158,857 frames are graded (the two halves).
- The ground truth is **mechanical, from corrected tracking**: `D` where the corrected tracking marks
  the ball dead, otherwise the team the tracking names as owner. Nothing is smoothed afterwards.
  So the target is a *provider convention*, not an intuitive notion of possession.
- Ranking metric `spellF1`: possessions are matched one-to-one by temporal IoU (thresholds 0.3-0.7).
  Count and boundaries matter, per-frame accuracy does not. 1 s late is about 0.65; differences
  below ~0.04 are noise.
- A possession is a maximal run of one team with the ball alive, so `H D H` is **two** possessions.
  Dead-ball detection therefore segments the match; it is as important as the team decision.
- `X` inside a possession is free on `spellF1` (forward-filled); at a hand-over it is costly.
- The ball is missing on 41% of graded frames, mostly in the decisive moments.
- Published floor: `baseline_smooth50`, spellF1 0.2433.

## 2. Constraints that shape the method

- No hand labels on the provided game (training or validation). External games: anything, if shared and documented.
- No use of the private GT, nor any human-produced data about the provided game (event data included).
- The 5 public minutes (17 possessions) check pipeline correctness only: no tuning, no model selection, no training.
- The repository must regenerate the submission file exactly. Data stays out of the repo (`data/` is gitignored).
- Terms: no provided data, or extracts of it, to hosted/cloud LLMs. This includes AI coding assistants.
  Practical rule for the team: assistants work on code and on external-game data only; outputs
  computed on the provided match stay local and are not pasted into assistant sessions. Confirm with
  the organisers if in doubt.

## 3. What public forks suggest (hypotheses, not templates)

Public forks are only part of what exists, none of their scores are private-set scores, and the
public window has 17 possessions (bootstrap interval about 0.3 wide). Treat every item as a
hypothesis to test on our own validation.

Common pattern: per-frame classifier on external matches + sequence decoder + separate dead-ball module.

| Observation (as claimed by the forks' own docs) | Our stance |
|---|---|
| Train on DFL/IDSSE (7 Bundesliga matches), sometimes also Metrica sample, SkillCorner, PFF | Plausible. Check licence and label definition first. |
| Degrade external matches to ~41% ball gaps (gap lengths from SkillCorner) | Adopt: it is the only way to see the real difficulty offline. |
| Leave-one-match-out tuning of decoder settings | Adopt. |
| LightGBM per-frame + Viterbi / min-duration decoder | Candidate baseline. |
| Mirrored training copies, boundary/motion features gave steady external gains | Test, per-match. |
| Dead-ball module: Law 9 (ball off pitch >1 s), restart shapes, still players | Candidate; tune the threshold on external data with a cap on live frames called dead. |
| Sequence NNs (GRU), 5-seed ensembles, wide hyperparameter search | Reported as no gain or worse on the real match. Low priority. |
| Video ball detector | Small, inconsistent gain, 80+ min GPU. Do not pursue. |
| Smart Tagging priors | No fork showed a reliable gain; one found a loss on the public window. Test by external ablation only. |
| External gains did not carry over to the competition window (one fork: ~0.69 external vs ~0.57 public; dead share 7% vs ~20%) | The main risk: domain gap. Plan for it explicitly (section 5). |

Smart Tagging exists only for the competition match. Forks that used it in training built stand-in
tags from DFL event files, timed and placed like the real ones (e.g. kick-off tag opens ~3.5 s before
the kick). This is the only way to train on tags; whether it helps is an open question.

Known pitfalls seen in forks: relabelling external ground truth with an invented minimum duration
(3 s) that is not part of the competition definition; selecting settings on the public window;
using the provided video for training (reported as banned by the organisers).

## 4. Hypotheses

- H1. The competition GT follows a provider convention similar to DFL's ball-status + possession fields.
  If so, external labels can be used as-is, without our own relabelling rule.
- H2. Most `spellF1` gain over the baseline comes from (a) correct dead-ball segmentation and
  (b) decoding that avoids splitting possessions at touches, not from model capacity.
- H3. A tree model on collective + kinematic features with temporal context, trained on degraded
  external matches, beats nearest-player + majority vote.
- H4. Smart Tagging helps mostly through restarts (dead-ball end/start), not through phase priors.

## 5. Plan

**Step 1. Understand the label convention.**
Load external matches, read provider ball-status and possession fields, compute run-length
distributions. Compare only to numbers already public (17 possessions in 5 min, ~20% dead frames,
mean possession ~14.2 s). Decide whether labels are used as-is.

**Step 2. Validation that does not touch the competition GT** (task 02).
External matches degraded to competition conditions (ball gaps ~41% with SkillCorner gap lengths,
dropped players, identity noise). Leave-one-match-out with the official `possession_metrics.py`,
bootstrap intervals. A change is accepted only if it improves the metric and intervals do not
overlap. The public window is used only to check alignment and format.

**Step 3. Baseline.** Per-frame gradient boosting -> P(H/A/D) -> decoder with a switch cost
(or minimum duration chosen by LOMO). Compare against `baseline_smooth50` on the same validation.

**Step 4. Dead-ball module** (task 03). Rules + learned detector; threshold set on external data
under a cap on live frames labelled dead. Track the dead share and the number of possessions.

**Step 5. Team under occlusion** (task 04). Team decision when the ball is missing, from team shape and
movement; evaluate by artificially hiding the ball in external matches.

**Step 6. Decoder** (task 05). Semi-Markov / Viterbi; `X` policy: abstain inside a possession when the
ball is lost, guess at transitions.

**Step 7. Smart Tagging ablation.** Real tags at inference; stand-in tags from DFL events for training.
Keep only if external ablation shows a gain.

**Step 8. Domain-gap checks** (label-free, on the competition match, run locally): possessions per
minute, dead share, share of frames abstained. Compare with external distributions, not with the
GT. Do not tune on them.

**Step 9. Freeze and deliver** (task 06). Deterministic run, pinned environment, validator, technical
document (approach, assets tried and dropped, training data with sources, possession rules,
limitations, reproduction). Name files `<team_name>.csv` / `<team_name>.pdf`.

## 6. Schedule

| Day | Goal |
|---|---|
| 13 Nov | external data, loader, validation harness, baseline on external data |
| 14 Nov | per-frame model + decoder |
| 15 Nov | dead-ball module, team under occlusion, tags ablation |
| 16 Nov | domain-gap checks, freeze configuration, reproducibility check |
| 17 Nov | final run, validator, technical document, form submission (before the deadline) |

## 7. Decision rules

- One change at a time; each logged in `docs/DECISIONS.md` including what was dropped.
- Accept only with improvement on external LOMO and non-overlapping bootstrap intervals.
- No setting derived from the competition match's ground truth. Competition-match tuning of any kind is out.
- Prefer the simpler variant when two are within noise (~0.04 spellF1).

## 8. Open questions

- Do the competition GT conventions match DFL's (H1)? Verify through distributions only.
- Are the external datasets' licences acceptable (DFL/IDSSE CC BY 4.0; Metrica sample licence unclear)?
- Terms on assistants: confirm with the organisers whether any aggregate output from the provided match may be shown to one.
- Team size (3-5 members) and who runs the code on the provided match.
