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
| `match_regime` | fitted by `scripts/regime_compare.py fit --write` (task 14): the ball regime of the degraded external matches read through version 1's labels (ball lost in play and in stoppages, where and how it is seen inside stoppages, placed balls, how it reappears, distance to the nearest player) must match the same statistics of the competition match, together with the agreement statistics of `match_like`; uses the optional mechanisms below | version 1's predicted labels of the match and our two pipelines' outputs; no labels, no published scores | evidence under the proposed rule C (docs/FUSION.md section 11); reported under rules A and B |

`label_free` understates the difficulty: the baselines score higher under it than on the real
match, and our two pipelines agree with each other more than they do on the match. `match_like`
and `published_baselines` are harsher and close to each other (ball offset 2.0 m against 1.5 m,
everything else equal), although they were fitted to different things. Since 2026-11-14 only
`label_free` and `match_like` count as evidence for a choice (docs/FUSION.md section 8):
`published_baselines` rests on numbers derived from the private ground truth, so it is printed
and disclosed and decides nothing.

Each file holds `params` (the keys of `validation.DEGRADATION_KEYS`, all required, plus any of
`validation.OPTIONAL_DEGRADATION_KEYS`), `source` (where the numbers come from) and, for a fitted
profile, `evidence` (the fit: residuals, the grid searched, the runners-up). To add a profile, add
a file; nothing else needs to change.

**Optional mechanisms** (task 14). A profile that does not name them degrades exactly as before:
their default switches them off and they draw from a second random stream, so `label_free`,
`match_like` and `published_baselines` give byte-identical degraded matches and the same profile
dicts as before they existed (`tests/test_degrade_mechanisms.py`; the feature cache and the
regime keys of trained models depend on both).

| key | default | what it does |
|---|---|---|
| `lead_in_fraction` | 0.5 | share of the live ball loss placed just before stoppages (the rest in free stretches of `free_gap_s`) |
| `placed_ball_prob` | 0 | share of restarts before which the ball is in view: first moving in a straight line from where it was last seen to the restart spot for `carried_ball_s`, then lying there for `placed_ball_s`, with tracker noise `placed_jitter_m` |
| `reacquire_s` | [0, 0] | after the other restarts, how long into play the ball is picked up (it then reappears already moving) |

Why they exist: `scripts/regime_compare.py table` compares the ball regime of every set with the
competition match through version 1's labels. The match loses the ball on about 8% of play and 75%
of stoppage time, and keeps it in view across the whole stoppage, often lying still before the
restart (a third of its stoppages); the older profiles lose it on 15-26% of play and 91-97% of
stoppage time, only at a stoppage's start, and the 2.0 m offset of `match_like` never lets it lie
still. `match_regime` is fitted to those statistics plus the agreement of `match_like`
(`scripts/regime_compare.py fit --write`).
