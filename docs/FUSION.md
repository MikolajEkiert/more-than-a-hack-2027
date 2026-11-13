# Combining the label-free track with a supervised model

Two tracks are being built: the label-free pipeline (`configs/unsup_v2.toml`) and a supervised
model trained on other matches. This document is the contract between them: what the supervised
track has to deliver, how the two are combined, and how the team decides between fusion and
choosing one output. Nothing here depends on how the supervised model is built.

## 1. Why fuse evidence, not labels

Both tracks end in the same decoder. The decoder takes, per frame, evidence that play is live
(`p_alive`) and evidence for the home team (`p_home`), and returns whole possessions.

- **Fusion happens before the decoder**: each stage's evidence is a weighted sum of the log-odds
  of its sources (`ball_state.blend`, `team.blend`). The output is still one coherent sequence.
- **Voting on two label files frame by frame is not an option.** Wherever the tracks disagree for
  a few frames it creates short possessions, and spellF1 counts every one of them.
- If only a finished label file is available from the supervised track, the choice is between
  the two files (section 4), not a fusion.

## 2. What the supervised track delivers

Either of these, per frame of a match:

| quantity | meaning | range |
|---|---|---|
| `p_alive` | probability that the ball is in play | [0, 1], no NaN |
| `p_home` | probability that the home team has the ball, given that it is in play | [0, 1]; NaN = no opinion |

or three class probabilities `p_h`, `p_a`, `p_d`, from which `p_alive = 1 - p_d` and
`p_home = p_h / (p_h + p_a)` are derived.

Two ways to plug them in:

1. **Registered methods (preferred).** `@register("ball_state", "<name>")` and
   `@register("team", "<name>")` in `src/possession/stages/`, taking a `Match` and returning the
   arrays above (`src/possession/AGENTS.md` has the contract). They then run on any match,
   including every degraded external match, and the repository regenerates the submission by
   itself. Model weights are either trained by a deterministic script in the repo or committed.
2. **Files.** One `.npz` (or `.npy`) per match, read by a source `{ file = "path/{game}.npz" }`.
   `{game}` is the match's name: `BAR-ATH` for the competition match, and
   `<GAME>@<profile>#<seed>` for a degraded external match (for example
   `METRICA-SG3@label_free#0`). `uv run possession export` writes the tracking of every validation
   condition under exactly those names (to `.cache/exports/`, no labels), so a model kept outside
   the repo can be run on the same inputs. Files are not committed; for the final submission the
   repo must still be able to regenerate them (RULES.md 3).

Requirements either way:

- **Leave-one-match-out.** Validation uses Metrica Sample Games 1-3. A score on one of them only
  counts if the model was trained without that match, in any degraded form. If the model trains on
  the Metrica games at all, it needs a way to train with one held out.
- **Label reading.** External labels exist in two readings, `touch` and `controlled`
  (`validation.CONVENTIONS`, docs/ARCHITECTURE.md). Say which one the model was trained on.
  `validation.load_external(game, convention)` returns the labels in either, and
  `validation.event_owner` is the derivation, so both tracks can use the same one.
- **Inputs it will see.** Automatic tracking: the ball missing for whole stretches of 5 s or more
  (41% of graded frames), about 20.5 players visible, no identities. `validation.degrade` makes an
  external match look like that; `possession inspect ball|players|gaps` shows the aggregates.
- **No fitting on the competition match** unless the team has cleared it against the organisers'
  email of 3 November (docs/research/UNSUPERVISED_RESEARCH.md section 6).

## 3. A fusion config

```toml
name = "fusion"

[ball_state]
method = "blend"
sources = ["stoppage_evidence", "supervised"]          # or { file = "probs/{game}.npz" }
weights = [1.0, 1.0]

[team]
method = "blend"
sources = ["touch_episodes", "supervised"]
weights = [1.0, 1.0]

[decode]
method = "switch_cost"
turnover_cost = 5.0
stop_cost = 12.0
restart_cost = 12.0
snap_start_s = 3.0
snap_end_s = 3.0
```

A source is a method name, a table `{ method = "...", <its parameters> }`, or a file table. A
weight of 2 counts a source twice; 0 silences it. For the team stage a source may abstain (NaN);
the result is NaN only where every source abstains, and the decoder then keeps the last owner.
`configs/blend_team_example.toml` is a working example with two of our own sources.

## 4. How to decide (fixed before measuring)

Everything is scored on the external matches, under both label readings and under both
degradation profiles (`possession validate ... --profile label_free|published_baselines`). The
public 5-minute window is a sanity check only and decides nothing.

1. Score the two tracks alone:
   `make validate CONFIGS="configs/supervised.toml" AGAINST=configs/unsup_v2.toml`.
2. Try fusion with a small grid of weights (0.5, 1, 2 for the supervised source, per stage) and
   re-tune the three switching costs once.
3. **Fusion is used only if** it passes the acceptance rule (`validation.compare`: mean gain of at
   least 0.02, no loss under either label reading, a win on at least two of three matches)
   against the better single track, under both profiles.
4. **Otherwise the better single track is used**, by the same rule. If neither beats the other by
   the rule, the label-free track is used: it has no training data to document and no domain gap.
5. Stages may be decided separately: for instance the supervised `p_home` with our `p_alive`.

What to expect, from measurements on the label-free track with one stage replaced by the truth
(game 3, `controlled` reading): a perfect team signal on frames with a tracked ball lifts 0.65
to 0.84; perfect stoppages to 0.73; a perfect team signal on frames without a ball to 0.73. The
team in control of a visible ball is where a supervised model can add most.

Two sources that read the same signal add nothing: blending our two team methods scores 0.589
against 0.598 for `touch_episodes` alone and is rejected by the rule. Fusion pays when the second
source has seen something the first has not.

## 5. Checklist for the report

For whichever combination is submitted, `docs/TECHNICAL_REPORT.md` section 3 lists every dataset
the supervised model was trained on (source, who labelled it, how much), and section 5 cites the
`experiments/runs/` files of the comparison above.
