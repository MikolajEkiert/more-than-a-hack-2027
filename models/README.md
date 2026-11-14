# Trained models

One file per trained model of a submitted configuration: `<key>.txt.gz` is a LightGBM model in
its text format, gzipped. `sha256` lists the hash of each unpacked file.

`<key>` is the cache key that `solution/submission.py:train_final` computes from everything that
defines the model: the feature code, the label function, the degradation parameters (calibrated
on unlabelled statistics of the competition tracking), the seeds and the LightGBM parameters.
`src/possession/supervised.py:seed_model_cache` unpacks these files into the package's cache
under that name, so a model is loaded only if it is exactly the one the current code would train.
If anything that defines the model changes, the key changes and the package trains again.

| File | Configuration | Trained on |
|---|---|---|
| `21853cf6e1215e719693.txt.gz` | `configs/hybrid_supervised.toml` (`labels = "raw"`) | 7 IDSSE matches (docs/TECHNICAL_REPORT.md section 3) |

To check a file against a fresh training: `MTAH_NO_CACHE=1 uv run possession run <config>` trains
from scratch, ignoring every cache, and must print the same sha256 for the submission file.
