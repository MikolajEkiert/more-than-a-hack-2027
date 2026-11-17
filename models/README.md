# Trained models

Every model that a submitted or fallback configuration loads is committed here, so the submission
file is regenerated without training. Each folder has a `sha256` file with the hash of every
unpacked model, and the code refuses a model whose hash does not match.

None of these models has seen a label of the competition match. Their training data is degraded to
resemble the competition tracking, with parameters calibrated on unlabelled statistics of that
tracking (docs/TECHNICAL_REPORT.md section 3).

| File | Used by | What it is | Trained on |
|---|---|---|---|
| `sequence/a2d3076d6d2c4fd1.npz` | version 4 (`configs/c14_best.toml`, submitted), `ball_state.sequence` | temporal network: is the ball in play (task 13), weights for the numpy inference | 7 IDSSE matches, both noise seeds, training seed 0 |
| `team/57471b9e1a31bec96010.txt.gz` | version 4 (`configs/c14_best.toml`, submitted), `team.supervised` with `features = "v1+team"` | LightGBM: which team has the ball, with our team features (task 14) | 7 IDSSE matches |
| `21853cf6e1215e719693.txt.gz` | version 1 (`configs/hybrid_supervised.toml`, branch `fallback/hybrid`) | LightGBM of the supervised package (`labels = "raw"`) | 7 IDSSE matches |

## How a model is found

- **LightGBM** (`*.txt.gz`). `<key>` is the cache key that `solution/submission.py:train_final`
  computes from everything that defines the model: the feature code, the label function, the
  degradation parameters, the seeds and the LightGBM parameters.
  `src/possession/supervised.py:seed_model_cache` unpacks the files into the package's cache under
  that name, so a model is loaded only if it is exactly the one the current code would train. If
  anything that defines the model changes, the key changes and the package trains again.
- **Temporal network** (`sequence/<key>.npz`). `src/possession/sequence.py` loads the network whose
  key matches its current definition and checks its sha256 against `sequence/sha256`.

## Retraining from scratch

- LightGBM: `MTAH_NO_CACHE=1 uv run possession run <config>` ignores every cache, trains again and
  must print the same sha256 for the submission file. With 8 threads (LightGBM's default on the
  machine we used) the retrained model is byte-identical to the committed one; with another number
  of threads it is not, because the histogram sums depend on it. That is why the models are
  committed.
- Temporal network: `uv run --with torch==2.15.0 possession train-sequence --folds --final`
  (about 1.5 hours on a CPU; docs/TECHNICAL_REPORT.md section 7).
