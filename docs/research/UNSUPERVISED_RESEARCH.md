# Research: an unsupervised solution for possession labelling

Status: research, 2026-11-13. Deadline 2026-11-17. Produced from five parallel literature/web
research passes plus a read of `evaluation/possession_metrics.py`. **No competition-match data was
read to write this.** Nothing here was tuned on the public 5-minute window.

Evidence tags used throughout:
**[V]** read at the source by a research agent · **[C]** computed by an agent from public
Metrica *sample* files (not our match) · **[P]** an unverified claim from another participant's
public repo · **[R]** recalled, not re-checked · **[D]** our own derivation or design judgement.

---

## 1. Recommendation in one page

Build a **label-free evidence model decoded by an explicit-duration state model (HSMM)**, with its
noise parameters fitted **on the match itself by EM** and its few structural hyperparameters chosen
**on other matches**. Concretely:

1. **Dead ball = classified ball gaps + visible-ball stoppage cues.** In Metrica's *corrected*
   tracking, which is what the ground truth is derived from, dead ball is literally "ball has no
   position" [C]. In our *automatic* tracking the ball is reportedly missing on ~77% of stoppage
   frames but only ~19% of live frames, with almost no gaps between 1 and 3 s [P]. So the unit of
   decision is the **gap**, not the frame: cluster gaps into "occlusion" and "stoppage" without
   labels, then fix the edges.
2. **Team = sticky, touch-based ownership.** Metrica's definition: in possession "as soon as they
   touch the ball"; not lost when an opponent's interference "does not have a significant impact"
   [V]. Evidence comes only from confirmed touches; everything else keeps the last owner.
3. **Decode with an HSMM over {H, A, D}**, emissions tied by home/away mirror symmetry so the
   states cannot swap, short fixed-count EM, Viterbi path, no `X` on graded frames.
4. **Align boundaries to the provider convention**, because timing is where most of the score
   between IoU 0.3 and 0.7 is decided (section 3).
5. **Optional second stage, only if it wins on external matches:** a player-only classifier
   trained on algorithmic pseudo-labels from this match, used as weak evidence inside long ball
   gaps. Never as a boundary detector.

Why unsupervised rather than the supervised plan in `EXPERIMENT_PLAN.md`: every fork that trained
on external matches reports that gains did not transfer (one: ~0.69 external vs ~0.57 public [P]),
and the largest open labelled set (DFL) follows a visibly different convention (7.8-9 turnovers
per live minute vs ~4.2 in the public window [P]). Fitting noise parameters on the match removes
the tracker-domain gap; taking the *convention* from Metrica's own documentation and sample data
removes the label-domain gap. No public participant repo we found does either with EM or
pseudo-labels: their "no training" entries are hand-set rule decoders [P].

What this is not expected to do: beat a careful rule decoder by a large margin on its own. The
published gain of sequence label models over majority vote ranges from +0.5 to +30 F1 and is
smallest when the simple vote is already good [V]. The value is robustness and a defensible,
general method.

---

## 2. What the ground truth actually is

| Fact | Source |
|---|---|
| GT is mechanical from Metrica's corrected tracking: D where the ball is dead, else the owning team | CHALLENGE.md |
| In Metrica Sample Game 3 (same EPTS format), all 101 restarts coincide with the **end** of a ball-NaN run, and all 86 stoppage events are followed one frame later by the **start** of one | [C] |
| Corrected Metrica ball position is "typically exactly equal to the position of the player who is currently in possession", partly reconstructed from annotated events | [V] Bischofberger et al. 2024 |
| Possession starts "as soon as they touch the ball" and lasts until another player takes over or the ball goes out | [V] Metrica event definitions |
| Possession "is not considered lost when an opposing player's interference does not have a significant impact" | [V] same |
| "Out of play: the ball went out of bounds or the referee has stopped play"; a set piece is "the player brings the ball back into play" | [V] same |
| The public Metrica EPTS sample has no ownership or alive channel; possession exists only in events | [V] |

Consequences for boundaries [D]:

- **H to A** happens at the new team's first *significant* touch, not after N seconds of control.
  A rule that waits for control must **backdate** to that first touch.
- **Alive to D** happens at the line crossing or the whistle, i.e. at or before the moment our
  tracker loses the ball. Participants report predicted stoppage starts 0.3-4.3 s **late** on 8 of
  8 public edges [P]. Bias early.
- **D to alive** happens at the restart kick or throw, not when the ball becomes visible again.
- **Short possessions are real**: 0.8 s and 1.3 s spells exist in the public window [P]. A hard
  minimum duration of 2-3 s deletes them; use evidence and switch costs instead.

### The dead share is probably not 20%

The 20% figure comes from the first 5 minutes only. Evidence that the full match is higher:

- Published ball-in-play share is 55-61%, i.e. 39-45% dead [V/snippet]; effective time is highest
  in the first 15 minutes (66.3%) and lowest in the last (55.9%) [V].
- If the ball is missing on 77% of dead and 19% of live frames [P], the published 41% missing
  rate implies a dead share of about 38%: `0.77 d + 0.19 (1 - d) = 0.41` [D].
- Participants' full-match outputs claim 32-50% dead [P]. The FAQ itself states 2,934 graded
  all-D frames before the second-half kickoff.

**Do not use 20% as a prior or a diagnostic target.** Plausible ranges for diagnostics [D]:
dead share 30-45%, 250-420 possessions, 90-110 stoppages (Siegle & Lames; Opta) [V/snippet].
Tracking-defined possession counts in the literature: 421 per match at mean 7.2 s (J-League,
touch-level) [V]; Metrica Sample Game 3 gives 343-409 depending on derivation [C][P].

---

## 3. What the metric rewards (from `possession_metrics.py`)

- Spells are matched **same-team only**, one-to-one, greedily by IoU; F1 is averaged over
  thresholds 0.3-0.7. Every true possession counts once, whatever its length.
- **Timing tolerance scales with spell length** [D]. For a spell of length L shifted by s,
  `IoU = (L - s) / (L + s)`:

  | IoU threshold | max shift | for a 5 s spell | for a 14 s spell |
  |---|---|---|---|
  | 0.3 | 0.54 L | 2.7 s | 7.5 s |
  | 0.5 | 0.33 L | 1.7 s | 4.7 s |
  | 0.7 | 0.18 L | 0.9 s | 2.5 s |

  With a median possession around 5-7 s, sub-second boundary placement decides the upper
  thresholds. A systematic bias (always late) is worse than noise.
- **Splits and merges cost about the same** [D]. A false blip of the other team inside a spell
  creates one or two false positives and damages the host's IoU. Missing a short opposing spell
  merges three true spells into one prediction: two false negatives. F1 is symmetric, so insert a
  spell when it is more likely real than not; there is no reason to be extremely conservative.
- **A missed stoppage between same-team spells** (`H D H`, e.g. a throw-in kept by the same
  team) merges two possessions into one. With ~100 stoppages per match, each bounding two spells,
  dead-ball edges touch well over half of all possessions.
- **`X` is never useful to us on graded frames.** It equals "repeat the previous label" for
  spellF1 and is a miss for the macroF1 tie-break. A state model always has a label. Emit no `X`
  inside the halves.
- One participant measured that a perfect ball in live play would add only about +0.03 [P]. If
  true, the score is in stoppage edges, convention and timing, not in recovering the ball.

---

## 4. The proposed method

It fits the existing three-stage contract. Stages return **per-frame evidence**, not smoothed
posteriors; smoothing happens once, in the decoder (otherwise evidence is counted twice).

### 4.1 `ball_state`: gaps and stoppage cues -> `p_alive`

Label-free cues, in order of expected value:

| Cue | How | Support |
|---|---|---|
| **Gap class** | Fit a 2-component mixture to log gap duration (plus the features below) per match; the valley between occlusions and stoppages is found automatically, so there is no hand-set threshold | bimodality claimed [P]; verify with `possession inspect ball` |
| Ball last seen near or beyond a line before the gap, or extrapolated across it | distance to boundary, velocity direction | [V] rule systems; Law 9 |
| Ball reappears at a restart location | corner mark (3 m), touchline (0.5 m), goal area, centre mark (2 m) with all players in own half | tolerances from Vidal-Codina et al. [V] |
| Visible but stationary ball | speed below ~0.5 m/s for more than ~2 s (free kicks, placed ball) | [D]; needed for the ~23% of dead frames with a visible ball [P] |
| Collective speed | 90th percentile of player speed; modelled as a per-state Gaussian learned by EM instead of a threshold | one fork uses p90 < ~2.1 m/s [P] |
| Smart Tagging `SET PIECES` | weak anchor for the restart time only | forks: coarse, one tag 12 s early [P] |

Edges [D]: D starts at the last ball sighting before a stoppage gap (or at the line crossing if
observed); D ends at the first confirmed kick after the ball reappears (ball acceleration of at
least 4 m/s² within ~2.5 m of a player [V]); a visible, stationary ball before that kick stays D.
A kickoff detector (each team in its own half, one player at the centre mark [V]) handles the
graded dead stretch before the second half as a general rule, with no frame constant.

Evidence that movement-only dead-ball models are weak: rule-based dead/alive was ~56% accurate in
one comparison [R]; a supervised player-only model reached 92% frames but with 1.7-1.8 s mean
edge error [V]; three forks switched such cues off [P]. Treat player motion as supporting
evidence, never as the edge detector.

### 4.2 `team`: touch-based sticky ownership -> `p_home`

1. Smooth the ball (Savitzky-Golay, order 2, 7 frames) and drop position jumps [V].
2. Candidate control: nearest player within 1.0-1.5 m of the ball. FIFA experts put the zone at
   0.5-1.5 m; automatic tracking wants the upper end [V].
3. Confirm a touch: ball direction change above 10°, speed change above 5 m/s, or acceleration of
   at least 4 m/s² during the in-zone run; otherwise it is a fly-by [V].
4. Duel: opponents of both teams within ~1 m gives no evidence [V].
5. Output evidence only on confirmed touches (soft, by distance margin between the nearest home
   and nearest away player); `NaN` elsewhere. The decoder keeps the last owner through passes,
   loose balls and gaps, which is exactly the provider's persistence rule.
6. Deflection veto [D]: a touch by the new team followed within ~1 s by the old team's touch or
   by a stoppage is not a change. This implements "interference without significant impact".

Known failure modes of these rules: fly-bys, crowded play, dribbles where the nearest player
flips, bounces, soft touches below the acceleration threshold [V]. Accuracy of published rule
systems drops from F 0.83-0.88 to 0.70 on independent data [V], which is why the thresholds'
*reliabilities* are learned on the match rather than assumed.

### 4.3 `decode`: mirror-tied HSMM

- **States** H, A, D. **Sequences**: each half separately.
- **Emissions**: the stage outputs, plus "ball visible" as its own Bernoulli emission per state
  (this is how missing-not-at-random is handled: missingness is evidence, and ball-derived cues
  contribute nothing when the ball is absent) [R].
- **Anchoring** so the states keep their meaning [D]: (1) tie parameters by symmetry,
  `P(vote=H | H) = P(vote=A | A)`; (2) clip every cue to be better than random after each
  M-step; (3) D anchored by gap class and restart geometry with a strong prior; (4) initialise
  from the hand-set rule decoder, never randomly.
- **Correlated cues**: all ball-derived cues are one source. Merge them into one categorical
  emission or temper them with weights below 1 (skweak does the latter) [V]. A naive model
  double-counts and becomes overconfident and flickery.
- **Why explicit durations**: at 25 fps a geometric duration with a 14 s mean is a switch penalty
  of only ~5.9 nats, which a few correlated frames override [D]. Use a duration distribution with
  a small minimum (about 0.5 s for possessions, so real short spells survive) and a heavy tail.
- **Fit**: 10-20 EM iterations, fixed count, both halves pooled, Dirichlet pseudo-counts.
  Deterministic by construction.
- **Decode**: Viterbi. Cost is `O((M D + M²) T)` [V]; decoding at 5 Hz and refining each boundary
  at 25 fps within ±0.5 s to the nearest touch or kick keeps it to seconds (estimate, unmeasured).
- **Implementation**: ~150 lines of numpy. No library fits: hmmlearn is in limited maintenance
  with no HSMM, pomegranate 1.0 lacks Viterbi and missing-data categoricals, hsmmlearn and pyhsmm
  are abandoned, dynamax and ssm bring JAX [V].
- **Hand-over across a live ball gap** [D]: place it at the extrapolated arrival time of the
  pre-gap ball if it was travelling, otherwise at the gap midpoint (minimum expected error).

Order of work: the same decoder with hand-set parameters and **no EM** is the first milestone and
the control. EM is kept only if it beats that control on external matches; its main risk is
drifting the states toward nuisance structure (ball visible vs missing, pitch zone).

### 4.4 Optional: player-only pseudo-label classifier

- Pseudo-labels: frames where the decoder's owner is confirmed by sustained clean control, with
  ~1 s trimmed at each end. Allowed by RULES.md ("pseudo-labels produced by an algorithm").
- Features: permutation-invariant, NaN-safe. Per-team centroid, **width-type shape** (lateral
  stretch, width, mean interpersonal distance), defensive line depth, mean velocity along the
  attack axis, home-minus-away differences, at offsets ±0.4-4 s. Augment by swapping teams.
- Model: LightGBM, deterministic settings, time-blocked validation (5-minute folds, 10 s gaps;
  random splits are meaningless on near-duplicate frames).
- Use: weak `p_home` evidence inside long gaps only.
- Evidence for: in possession a team is ~13 m wider (+34%) with 56% more area, Cohen's d ≈ 1.9-2.0
  for width-type measures (phase averages, one team, one match) [V]; deep player-only models
  reach 85-92% team accuracy on clean data [V].
- Evidence against: shape **lags** turnovers (just-won shape is indistinguishable from defending
  shape) [V]; one fork found team shape on every frame merges possessions [P]; visible-ball frames
  differ systematically from hidden-ball frames, and semi-supervised methods degrade under that
  shift [V]. Check with a "visible vs hidden" domain classifier and synthetic gaps.

### 4.5 Rejected

| Idea | Why |
|---|---|
| Learned ball imputation (Ballradar, TranSPORTmer, PathCRF weights) | 2-4 m error, needs clean 22-player tracking, GPUs and many matches; nobody has trained one self-supervised on a single match [V] |
| Broadcast ball as a gap filler | adds false turnovers [P]; unsynchronised timeline |
| Smart Tagging as a team prior | 44-100% agreement per code, phases span turnovers [P] |
| Generic change-point detection (PELT etc.) | returns unlabelled segments; equivalent to a switch penalty without state semantics [D] |
| Training on raw DFL labels | different convention, too fragmented [P] |
| Hard minimum possession of 2-3 s | deletes real sub-2 s spells [P] |

---

## 5. Validation without touching the rules

1. **Synthetic unit tests** per method (existing convention).
2. **Other matches, same procedure.** Run the identical fit-on-itself pipeline on a labelled
   external match and score with the official scorer. This tests exactly what the organisers ask
   for: a method that works on the next match.
   - **Metrica sample-data, Game 3 first** (same provider, same EPTS format): D = ball-NaN runs
     longer than a few frames; owner = team of the last on-ball event, `BALL LOST` staying with
     the losing team until the opponent's next on-ball event [C]. No licence file; download at
     run time, do not redistribute, credit the source [V]. Games 1-2 as extra folds.
   - **Degrade it to our conditions**: remove the ball on ~19% of live frames in short gaps and
     restore it on ~23% of dead frames, drop players to ~20.5 visible, fragment tracks. Take the
     gap statistics from `possession inspect ball` aggregates, not from the [P] figures.
   - DFL (CC-BY 4.0, explicit per-frame flags) only as a secondary check, and only after
     harmonising its sub-2 s sandwiches [P]; SkillCorner (MIT, ~60% ball detection) is a
     realistic-noise alternative.
3. **Label-free diagnostics on the competition match** (aggregates only): possession count, dead
   share, share of spells under 1 s, number of stoppages, and **cross-half stability** (fit on
   half 1, decode half 2 and the reverse, score prediction against prediction).
4. **Public 5 minutes: one sanity look**, recorded, never used to choose. With 17 possessions the
   bootstrap interval is ~0.3 wide.

Hyperparameters that EM cannot learn (emission tempering, duration-prior shape, control radius)
are chosen in step 2 only. Accept a change only when it beats the noise band (±0.04).

---

## 6. Risks and open questions

1. **Rules check, needs a human.** A participant reports an organiser email of 3 November saying
   that nothing may be trained, fitted or calibrated on the ground truth or the provided video,
   and that tunables must come from external data [P]; we have not seen the original. RULES.md
   explicitly allows unsupervised methods and algorithmic pseudo-labels on this game, and EM here
   uses tracking only. Read the email's exact wording. Hedge: a config switch that freezes the
   parameters fitted on external matches instead of fitting on the match.
2. **The gap statistics are someone else's claim.** The whole dead-ball design leans on gap
   length being bimodal on our match. Check it first; if it is not, section 4.1 falls back to
   restart geometry plus stationary ball.
3. **Whistles are not observable.** Stoppages for fouls with a visible ball will start late.
4. **Hand-overs happen where the ball is missing** (duels, aerial balls). Timing there rests on
   interpolation, not evidence.
5. **Degradation realism.** A noise simulator that is too kind makes external validation
   optimistic, which is the failure the forks report.
6. Not found by any search: a decoder derived for IoU-matched segment F1; a per-frame accuracy
   for "the wider team has the ball"; a published ball-free dead-ball detector; a published
   gap-length distribution for automatic ball tracking; Metrica's own specification of per-frame
   ownership.

---

## 7. Four-day plan

| Day | Work | Maps to |
|---|---|---|
| 14 Nov | External harness: Metrica Game 3 labels + degradation. Gap classifier. Touch-based team evidence. Hand-set switch-cost decoder (the control). | tasks 02, 03, 04 |
| 15 Nov | HSMM with mirror-tied EM; boundary alignment (backdating, early D start, restart kick, kickoff rule). Ablations on external matches. | task 05 |
| 16 Nov | Optional pseudo-label classifier, kept only on an external win. Diagnostics on the competition match. Freeze the config. | task 04 |
| 17 Nov | Reproducibility run, validator, technical report, submit. | task 06 |

Each row needs its task file updated before code is written (`/plan-task`), and the method choice
belongs in `docs/DECISIONS.md` once the team agrees to it.

---

## 8. Relation to the team's other research (PR #2, `research/RESEARCH.md`)

PR #2 is a general survey written independently, without reading other teams' work, and it leans
towards supervised models trained on external matches. The two documents agree on most of the
picture and differ on three points that change the design.

Agreement: the three-way split of the problem; boundary timing as the main cost; HSMM / segmental
decoding instead of a majority window; touch-based ownership with a change of the ball's velocity
as the touch signal; explicit `D` before the second-half restart; pseudo-labels as an option;
Metrica sample data as the place to learn the provider convention. PR #2 also adds a synthetic
sensitivity simulation of the metric, which this document does not have (its absolute values are
generator-dependent, as its review notes).

Differences:

| Topic | PR #2 | This document |
|---|---|---|
| DFL / IDSSE labels | "the same semantics as the GT" | A different convention: 41-52% dead and 7.8-9 turnovers per live minute, against ~4.2 in the public window [P]. Usable only after harmonisation, and second to Metrica sample data. |
| How dead ball is encoded | a provider flag | In Metrica's corrected tracking it is the ball having no position [C], which makes gap classification the core of dead-ball detection. |
| Where parameters come from | supervised training on degraded external matches | Noise parameters fitted on the match without labels; only structural hyperparameters from external matches. Avoids the transfer failure the forks report. |

PR #2 also proposes whistle detection from audio and video ball detection. Both are outside this
document's scope; note the reported organiser clarification of 3 November (section 6) before
fitting anything on the provided video.

---

## 9. Sources

Rule-based possession and events
- Vidal-Codina et al. 2022, automatic event detection from tracking: https://ar5iv.labs.arxiv.org/html/2202.00804
- Link & Hoernig 2017, individual ball possession: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953
- Bischofberger et al. 2024: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298107
- Morra et al., SoccER: https://ar5iv.labs.arxiv.org/html/2004.04147
- ETSY synchronisation: https://lirias.kuleuven.be/retrieve/731472
- databallpy possession implementation: https://github.com/Alek050/databallpy

Sequence models and decoding
- skweak (Lison et al. 2021): https://ar5iv.labs.arxiv.org/html/2104.09683
- FlyingSquid: https://ar5iv.labs.arxiv.org/html/2002.11955
- Linked HMM / WISER: https://ojs.aaai.org/index.php/AAAI/article/view/6009
- Explicit-duration forward-backward (Yu & Kobayashi 2003): https://ieeexplore.ieee.org/document/1172820
- Sticky HDP-HMM (Fox et al.): https://arxiv.org/abs/0905.2592
- MS-TCN: https://arxiv.org/abs/1903.01945 · ASRF: https://arxiv.org/abs/2007.06866
- Cross-validated likelihood for HMMs (Celeux & Durand 2008): https://link.springer.com/doi/10.1007/s00180-007-0097-1

Ball and possession from players
- Kim et al. 2023 (Ballradar): https://ar5iv.labs.arxiv.org/html/2306.08206
- TranSPORTmer: https://arxiv.org/html/2410.17785v2
- PathCRF: https://arxiv.org/html/2602.12080
- Lang et al. 2022, stoppages from player tracking: https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/
- Oliver et al. 2018, semi-supervised under class shift: https://arxiv.org/abs/1804.09170
- Arazo et al. 2020, confirmation bias: https://arxiv.org/abs/1908.02983

Collective behaviour and priors
- Zhang 2022 (LJMU thesis), team shape by possession phase: https://researchonline.ljmu.ac.uk/id/eprint/19091/1/2022guangzezhangmphil.pdf
- J-League possession statistics: https://ar5iv.labs.arxiv.org/html/2308.05460
- Opta stoppage durations: https://theanalyst.com/articles/premier-league-delays-set-piece-throw-in-stats
- Effective playing time by period: https://jssm.org/jssm-17-662.xml-Fulltext
- Tojo et al. 2023, stoppages: https://pmc.ncbi.nlm.nih.gov/articles/PMC10442814

Data and conventions
- Metrica sample-data and event definitions: https://github.com/metrica-sports/sample-data
- Bassek et al. 2025 (DFL): https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/
- SkillCorner opendata: https://github.com/SkillCorner/opendata
- Organiser repository: https://github.com/metrica-sports/more-than-a-hack-2027

Participant repositories read for ideas only (all claims unverified): danieltyukov/more-than-a-hack-2027,
PetricaNancu/ball-possession-barca-hackathon, bassem97/syntaxeror-mtah2027,
suleyman416/more-than-a-hack-2027, Ratnesh-101/more-than-a-hack, novo-official/mtah2027-possession.
