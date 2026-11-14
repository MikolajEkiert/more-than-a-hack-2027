# Degradation profiles

The external matches (Metrica sample games) are clean, corrected tracking. Before a pipeline is
scored on them, `validation.degrade` makes them look like the automatic tracking of the
competition match. How much it degrades is a **profile**: one JSON file here, chosen by name.

```bash
uv run possession validate configs/x.toml                               # label_free (default)
uv run possession validate configs/x.toml --profile match_like
make validate CONFIGS="configs/x.toml" PROFILE=match_like
```

| profile | how its numbers were set | uses | role |
|---|---|---|---|
| `label_free` | by hand, so that aggregates of the degraded tracking match `possession inspect gaps` on the competition tracking (share of frames with a ball, long ball gaps, players visible, smoothness of the ball, ball-to-player distance and relative speed) | no labels, no published scores | evidence |
| `match_like` | fitted by `possession calibrate --target match --write`: on the degraded external matches our two pipelines (`hybrid_supervised`, `unsup_v2`) must agree with each other as they do on the competition match (possessions, team, how sure the team evidence is) | outputs of our own pipelines on the match; no labels, no published scores | evidence |
| `published_baselines` | fitted by `possession calibrate --write`: the four organiser baselines must score on the degraded external matches what EVALUATION.md publishes for them on the full match | the eight published numbers, which are computed from the private ground truth | reported only |

`label_free` understates the difficulty: the baselines score higher under it than on the real
match, and our two pipelines agree with each other more than they do on the match. `match_like`
and `published_baselines` are harsher and close to each other (ball offset 2.0 m against 1.5 m,
everything else equal), although they were fitted to different things. Since 2026-11-14 only
`label_free` and `match_like` count as evidence for a choice (docs/FUSION.md section 8):
`published_baselines` rests on numbers derived from the private ground truth, so it is printed
and disclosed and decides nothing.

Each file holds `params` (the keys of `validation.DEGRADATION_KEYS`, all required), `source`
(where the numbers come from) and, for a fitted profile, `evidence` (the fit: residuals, the
grid searched, the runners-up). To add a profile, add a file; nothing else needs to change.
