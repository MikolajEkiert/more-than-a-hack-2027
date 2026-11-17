# What other teams report (public repos, snapshot 2026-11-13)

Source: forks of metrica-sports/more-than-a-hack-2027 (288 listed; 80 with their own commits) and
92 stand-alone repos found by GitHub search. All figures are **self-reported** in READMEs, reports
and commit messages. Nobody outside the organisers can compute the full-match score, because the
ground truth is private. Read every number below with its protocol.

## The three kinds of number

| protocol | what it is | how far to trust it |
|---|---|---|
| **public 5-min window** | 17 possessions of this match (the one GT file we have) | noise about ±0.11-0.18. EVALUATION.md shows it picks the wrong baseline. Anything trained or tuned on it is meaningless, and training on it breaks RULES.md 1-2 |
| **other games** | open tracking data (DFL Bundesliga, IDSSE, Metrica sample-data, PFF, SkillCorner) with labels derived by an algorithm | the only honest proxy for the hidden score. Still a different data source than this match's ATD |
| **full match** | the real ranking | only the organisers have it |

## Reported results

Other-game validation (the credible group), best first:

| spellF1 | where | note |
|---|---|---|
| 0.857 | 7dracoder, one external match | single match; same repo states the ceiling is about 0.90 |
| 0.681 | agamemnon-labs, several held-out matches (worst 0.585, best 0.774) | Smart Tagging is worth about +0.09; without it 0.62. First version (2 logistic regressions): 0.501 |
| 0.62-0.68 | Derekko-web, development matches (DFL/PFF/SkillCorner, plus Metrica samples) | most rigorous log: fixed acceptance gates, 20+ numbered experiments |
| 0.624 | AndriesDKock, 10 held-out matches (3 Metrica + 7 DFL) | rule fixed in advance: +0.02 mean and better on at least 7/10 matches |
| 0.595 | Spyyy004, Metrica games | |
| 0.54 | AdrianBar812, held-out DFL | up from 0.20 after tuning |
| 0.46-0.54 | Wichai-pan, ZainnB (pooled 0.46) | |

Public 5-minute window only (weak evidence): typical 0.42-0.58 (e.g. 0.4235, 0.4457, 0.457, 0.4556,
0.4875, 0.5, 0.57-0.581). The organisers' own baselines score 0.3256 (smooth50) and 0.4242 (smooth125)
there, so most of these are within noise of a baseline.

Unverifiable or not credible:
- 0.9375 (debjyotydasgupta): supervised training **on the public GT slice and scored on the same
  slice**. Pure overfitting, and against RULES.md.
- 0.7733 with macroF1 0.9524 (joynard commit message) and 0.64 with macroF1 0.85-0.87 (Camgomocod
  commit messages): protocol not stated. Values of that size are only plausible on the public window.
- ZainnB reports a "human ceiling" 0.894 from its own blind re-annotation of the public window.
  Hand labels on the provided game are not allowed for validation either (RULES.md 1).

## What the better teams do (ideas worth taking, within the rules)

1. **Train and validate on other games**, with possession labels derived by an algorithm from open
   data (DFL, IDSSE, Metrica sample-data, PFF). Rule: hand labels on *other* games are fine, but
   must be disclosed. This is our tasks/02.
2. **Smart Tagging as a feature**: +0.09 spellF1 (agamemnon-labs), +0.03 for touch features
   (AndriesDKock).
3. **Dead ball is the biggest lever**: an oracle D mask adds +0.07-0.09 (jekkiechan); kinetic
   "everyone at rest" filters and lead-lag detectors raise D quality (Camgomocod, ZesterJ). Our tasks/03.
4. **A structured decoder** (3-state Viterbi / HMM / hysteresis with minimum durations) rather than a
   window vote. Our tasks/05.
5. **Ownership is the residual error**: with true ownership substituted, Derekko-web's score goes
   0.68 -> 0.89 (7dracoder: 0.857 -> 0.90 even with ground-truth boundaries). Our tasks/04.
6. Repeatedly stated lesson: **selection rules fixed in advance** and a noise band of about ±0.04;
   several teams report that the public window preferred the wrong variant.

## Reading it honestly

A defensible target on the hidden score is roughly **0.55-0.70** (the other-game range), versus
the baseline 0.2433. A result above 0.75 reported without an external-game protocol should be
assumed to be public-window overfitting.
