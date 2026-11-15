# Approach A (unsupervised + heuristics): proposed revisions

Status: proposal, 2026-11-13. Theory only: nothing here was measured. Compares Approach A with
the plan on branch `research/unsupervised` (`docs/research/UNSUPERVISED_RESEARCH.md`, below "the
branch plan") and proposes changes to A. Decisions belong in `docs/DECISIONS.md` once the team
agrees.

## 1. Verdict

Approach A is sound, and about 80% of it overlaps with the branch plan. Both use rule-based
evidence, a touch-based ownership rule (Vidal-Codina et al.), an HSMM over {H, A, D} and optional
algorithmic pseudo-labels. A differs in emphasis, and three of those differences would cost score:

1. Dead ball rests on player motion, not on gaps in the ball track.
2. Reliability of each signal is estimated by a separate Dawid–Skene stage before the decoder.
3. Ball-free signals (team shape, advance, pressure, Smart Tagging phases) are first-class
   evidence.

A also adds one idea the branch plan lacks and should adopt: **restart rules as a constraint on
the state after a stoppage.**

Recommendation: keep A's structure and goals, apply the changes in section 3, and borrow the
validation answer from the branch plan, because that answer resolves A's own stated weakness
("dozens of thresholds, no honest way to set them").

## 2. Side by side

| Component | Approach A | Branch plan | Proposal |
|---|---|---|---|
| Dead-ball core | ball out of pitch, collective slow-down (Lang et al.), restart layouts, Smart Tagging windows | **gap classification**: in Metrica's corrected tracking, from which the GT is derived, dead ball is "ball has no position" | gaps as the core; A's cues as supporting evidence and for restart edges |
| Possession change | controlled touch (Vidal-Codina) | same, plus a deflection veto and backdating to the first touch | same as the branch plan |
| Ball missing | team shape, advance, pressure, Smart Tagging phases | weak evidence inside long gaps only | "which team" inside gaps, never "when" |
| Combining signals | Dawid–Skene / Snorkel, then HSMM | reliabilities as HSMM emission parameters, fitted by EM | inside the HSMM, not as a separate stage |
| Self-training | pseudo-labels, model predicts through occlusions | optional, kept only if it wins on an external match | optional, same gate |
| Decoder | HSMM with typical durations | HSMM, mirror-tied, short minimum duration, heavy tail | branch plan, plus A's restart constraint |
| Restart constraint | throw-in, corner, goal kick to the team opposite the last touch; kickoff to the conceding team | none | **adopt**, as a soft transition preference |
| Boundary convention | not addressed | backdate hand-overs, start D early, end D at the restart kick | adopt |
| Setting thresholds | "no honest way" | EM on the match without labels; structural hyperparameters on an external match | adopt |

## 3. Proposed changes

### 3.1 Dead ball: make the ball gap the unit of decision

**Change.** Classify every gap in the ball track as an occlusion or a stoppage, using a two-component
mixture on gap length and context (last ball position relative to the lines, player speed, restart
layout at the reappearance). Then place the edges.

**Why.**
- The ground truth marks D where corrected tracking has no ball position, so the gap structure is
  the most direct label-free signal available.
- The 92% of Lang et al. comes from a **supervised** model, so an unsupervised version will not
  reach it. Even at 92% frame accuracy, their edge error is 1.7–1.8 s. At IoU 0.5–0.7 a 5 s
  possession tolerates only 0.9–1.7 s of shift, so motion-defined edges lose most of the upper
  thresholds.

**What stays from A.** Collective slow-down, ball leaving the pitch and restart layouts are
evidence for the gap classifier and the edge refinement, not the edge detector itself. Smart
Tagging SET PIECES windows anchor only the restart time, which is then found exactly in tracking.

**Check first.** That gap lengths are bimodal on our match (`possession inspect ball`, aggregates
only). If they are not, fall back to restart layout plus a visible, stationary ball.

### 3.2 Signal reliability: Dawid–Skene inside the HSMM

**Change.** Do not run a separate per-frame label model. Make each heuristic an emission of the
HSMM with its own confusion parameters, fitted by EM on the match. This is Dawid–Skene with time:
a linked-HMM label model.

**Why.**
- Dawid–Skene assumes the labelling functions err independently. Here they do not. All
  ball-derived cues are effectively one source, and shape, advance and pressure are strongly
  correlated. The model would read their agreement as confidence and become overconfident and
  flickery.
- A per-frame label model followed by an HSMM smooths twice and counts the same evidence twice.
- H and A are symmetric, so an unsupervised model can swap them.

**How to keep it honest.**
- Merge correlated cues into one categorical emission, or temper them with weights below 1.
- Tie parameters by home/away mirror symmetry, so that P(vote H | H) = P(vote A | A).
- Initialise from the hand-set rule decoder, not randomly.
- Use a fixed, small number of EM iterations.

This keeps A's intent: Smart Tagging and every heuristic get a principled weight.

### 3.3 Ball-free evidence: "which team", never "when"

**Change.** Use team shape (Shaw & Glickman), the teams' advance toward goal and pressure around
the likely carrier only as `p_home` evidence **inside long ball gaps**. They must not move a
boundary that touch evidence has placed.

**Why.** Shape lags turnovers: a team that has just won the ball still looks like the defending
team. Used at boundaries, these signals push hand-overs later, which is the most expensive error
for spellF1. One public fork reports that frame-level shape evidence merged possessions.

**Smart Tagging team phases: drop as team evidence.** Phases span turnovers (reported agreement is
44–100% depending on the code), and a long interval acts like hundreds of correlated votes. Keep
Smart Tagging for restart timing only.

### 3.4 Restart rules: adopt, as a soft constraint

**Change.** Split the dead state into "D after H" and "D after A". Both have identical emissions
and both output `D`; they differ only in their exit transitions:

| Restart | Next team | Strength |
|---|---|---|
| throw-in, corner, goal kick | opposite of the last touch | strong |
| kickoff after a goal | conceding team | strong (whoever scored had the last possession) |
| kickoff at the start of a half | from the kickoff layout | strong |
| free kick, penalty | the fouled team, often the same as before the stop | weak; leave to the evidence |
| drop ball, injury stop | usually the team that had the ball | weak |

**Why.** By the branch plan's count, dead-ball edges touch well over half of all possessions. A
wrong team after a stoppage costs a whole possession, and a missed `H D H` merges two spells into
one. This constraint adds no new thresholds: the exit preferences are transition parameters that
EM can fit.

**Limits.**
- "Last touch" is uncertain with automatic tracking, and the restart type must first be detected
  (from the layout, section 3.1).
- Make it a soft preference (for example, a starting point of 70/30 that EM refines), never a hard
  rule.

### 3.5 Boundary convention (added)

Place boundaries the way the ground truth does:
- **Hand-over:** at the new team's first significant touch. A rule that waits for control must
  backdate to that touch.
- **Start of D:** at the line crossing or the last sighting before a stoppage gap. Participants
  report predictions 0.3–4.3 s late on all 8 public edges.
- **End of D:** at the restart kick, not at the ball's reappearance. A visible, stationary ball
  stays D.
- **Short spells are real** (0.8 s and 1.3 s exist in the public window). Use a small minimum
  duration, about 0.5 s, not 2–3 s.

### 3.6 Self-training: keep, behind a gate

Keep A's pseudo-label model as an optional second stage, but:
- take pseudo-labels only from sustained, clean control, and trim about 1 s at each end;
- use the model only as weak `p_home` evidence inside long gaps;
- accept it only if it beats the no-self-training version on the external match by more than the
  noise band.

**Why the gate.** Pseudo-labels exist where the ball is visible, but the model is needed where it
is hidden. Distribution shift plus confirmation bias is the standard way self-training fails.

### 3.7 Answer to "no honest way to set the thresholds"

- **Reliabilities, noise rates and transition preferences:** fitted by EM on the match, with no
  labels (allowed by RULES.md: unsupervised learning on the provided game).
- **Structural hyperparameters** (gap-mixture form, emission tempering, duration-prior shape,
  control radius, restart-preference strength): chosen on the external harness, Metrica Sample
  Game 3 with degraded tracking. The harness exists: `unsup_control` scores 0.503 there against
  0.321 for the baseline.
- **Acceptance rule, fixed in advance:** a change stays only if its gain on the external match
  exceeds ±0.04 spellF1.
- **Public 5-minute window:** one sanity look per version, never used to choose.

## 4. Revised Approach A, in one list

1. Ball gaps classified as occlusion or stoppage, with A's cues as features.
2. D edges from the last sighting or line crossing, and from the restart kick. Restart layouts
   identify the restart type and the kickoff.
3. Controlled-touch ownership with a deflection veto, backdated to the first touch.
4. Ball-free evidence (shape, advance, pressure) for the team only, inside long gaps.
5. HSMM over {H, A, D-after-H, D-after-A}:
   - heuristics are emissions whose reliabilities are fitted by EM;
   - parameters are mirror-tied;
   - durations have a short minimum and a heavy tail;
   - restart preferences are soft transitions.
6. Optional: a pseudo-label model inside gaps, kept only if it wins on the external match.
7. Structural choices are made on the external match, with the acceptance rule fixed in advance.

## 5. Risks

- **Rules.** A participant cites an organiser email (not seen by us) saying nothing may be fitted
  on the match. EM and self-training on the match are exposed. Hedge: a config switch that freezes
  the parameters fitted on the external match, and a question to the organisers before relying on
  it.
- **Gap bimodality** is assumed, not verified.
- **Restart-type detection** feeds the restart constraint, so its errors compound. Keep the
  constraint soft.
- **Whistles are not observable.** Fouls with a visible ball will start D late.
- **The external harness may be too kind.** Degradation realism decides whether choices made there
  transfer.

## 6. Order of work (deadline 2026-11-17)

1. Verify gap bimodality with aggregates. Build the gap classifier and the D edges. (task 03)
2. Touch ownership with backdating and the deflection veto. (task 04)
3. HSMM with split D states and soft restart transitions, with hand-set parameters first as the
   control. (task 05)
4. EM for the reliabilities, kept only if it beats the control on the external match.
5. Ball-free team evidence inside gaps, same gate.
6. Optional self-training, same gate. Freeze the config, then the reproducibility run. (task 06)

## Sources

As in `docs/research/UNSUPERVISED_RESEARCH.md` §9: Vidal-Codina et al. 2022, Link & Hoernig 2017,
Lang et al. 2022, Dawid & Skene 1979 / Snorkel / skweak, linked HMM (WISER), Yu & Kobayashi 2003
(explicit-duration HMM), Shaw & Glickman (team shape), Oliver et al. 2018 and Arazo et al. 2020
(failure modes of self-training).
