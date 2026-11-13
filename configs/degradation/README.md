# Degradation profiles

The external matches (Metrica sample games) are clean, corrected tracking. Before a pipeline is
scored on them, `validation.degrade` makes them look like the automatic tracking of the
competition match. How much it degrades is a **profile**: one JSON file here, chosen by name.

```bash
uv run possession validate configs/x.toml                               # label_free (default)
uv run possession validate configs/x.toml --profile published_baselines
make validate CONFIGS="configs/x.toml" PROFILE=published_baselines
```

| profile | how its numbers were set | uses |
|---|---|---|
| `label_free` | by hand, so that aggregates of the degraded tracking match `possession inspect gaps` on the competition tracking (share of frames with a ball, long ball gaps, players visible, smoothness of the ball, ball-to-player distance and relative speed) | no labels, no published scores |
| `published_baselines` | fitted by `possession calibrate --write`: the four organiser baselines must score on the degraded external matches what EVALUATION.md publishes for them on the full match | the eight published numbers (documentation, never the ground-truth files) |

`published_baselines` is harsher: with `label_free` the baselines score higher on the external
matches than on the real one, so that profile understates the difficulty. Whether calibrating on
published scores is acceptable is the team's call (docs/DECISIONS.md); both profiles are kept so
the choice is one switch, and results are reported under both.

Each file holds `params` (the keys of `validation.DEGRADATION_KEYS`, all required), `source`
(where the numbers come from) and, for a fitted profile, `evidence` (the fit: residuals, the
grid searched, the runners-up). To add a profile, add a file; nothing else needs to change.
