"""The supervised track's model (package `solution/`) as evidence for the pipeline's stages.

`solution/` is the supervised branch, merged as written: seven Bundesliga matches of open
corrected tracking (IDSSE, Bassek et al. 2025, CC-BY 4.0) are degraded to look like automatic
tracking, turned into hand-made features and used to train one LightGBM with three classes,
H / A / D. This module is the only place of the pipeline that calls it. It

  * converts a `Match` into the layout that package works on (`container`)
  * trains the package's final model, once (`model`); the package caches it on disk
  * returns the three class probabilities per frame for any match (`probabilities`)

The stage methods `ball_state.supervised` and `team.supervised` read those probabilities; the
decoder stays ours (docs/FUSION.md section 6).

Two things a reader should know:
  * The degradation applied to the training matches is calibrated on unlabelled statistics of
    the competition match (share of frames without a ball, mean length of those stretches,
    players visible), exactly as in the supervised branch. So the model needs the match data
    even when it scores another match, and it is trained for this match's tracking.
  * Nothing is stored in data/. The open data (2.7 GB, downloaded on first use) goes to
    .cache/idsse/, the package's cache of degraded matches, features and the trained model to
    .cache/solution/. Both directories can be deleted; everything is rebuilt, identically.
  * The trained models of the submitted configurations are committed in models/ (gzipped
    LightGBM text, with their sha256). They are copied into the package's cache under the
    package's own cache key, so a model is used only if it is exactly the one this code, these
    parameters and this match's statistics would train; otherwise the package trains as usual.
    With a committed model the open data is not downloaded at all. `MTAH_NO_CACHE=1` makes the
    package ignore every cache and train from scratch, which is how the committed file is
    checked (docs/FUSION.md section 7).
"""

import gzip
import hashlib
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np

import possession_metrics as pm
from possession import features
from possession._paths import CACHE_DIR, REPO_ROOT
from possession.match import Match

IDSSE_DIR = CACHE_DIR / "idsse"
PACKAGE_CACHE_DIR = CACHE_DIR / "solution"
MODELS_DIR = REPO_ROOT / "models"  # <cache key>.txt.gz, and `sha256` listing each model's hash


def _package():
    """The modules of solution/ this module uses, with their storage pointed at .cache/.

    The directories are read from the environment when the package is first imported, so they
    are set here, before the import, and nowhere else imports the package.
    """
    os.environ.setdefault("MTAH_IDSSE_DIR", str(IDSSE_DIR))
    os.environ.setdefault("MTAH_CACHE_DIR", str(PACKAGE_CACHE_DIR))
    from solution import degrade, labels, submission
    from solution import features as package_features

    return degrade, labels, submission, package_features


def container(match: Match) -> dict:
    """`match` in the layout of solution/container.py: float32 positions in metres, no labels."""
    return {
        "px": match.players[:, :, 0].astype(np.float32),
        "py": match.players[:, :, 1].astype(np.float32),
        "team": match.team_of_track.astype(np.int8),
        "bx": match.ball[:, 0].astype(np.float32),
        "by": match.ball[:, 1].astype(np.float32),
        "half": match.half.astype(np.int8),
        "label": np.full(match.n_frames, pm.X, np.int8),
        "pitch": np.asarray(match.pitch, np.float32),
        "name": match.game,
    }


def label_function(spec: str):
    """The package's label function named by `spec`: "raw", "tau<seconds>", "controlled<seconds>".

    "raw" is the provider's own flags and the supervised track's choice. "tau2" merges a spell of
    the other team shorter than 2 s, between two spells of one team, into them (a touch, not a
    possession); "controlled2" lets a hand-over count only if the new team keeps the ball 2 s.
    """
    _, labels, _, _ = _package()
    for prefix, make in (("tau", labels.harmonised), ("controlled", labels.controlled)):
        if spec.startswith(prefix):
            try:
                seconds = float(spec[len(prefix) :])
            except ValueError:
                break
            return make(seconds)
    if spec == "raw":
        return labels.raw
    raise ValueError(f"unknown labels {spec!r}; use 'raw', 'tau<seconds>' or 'controlled<seconds>'")


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def seed_model_cache(models_dir: Path = MODELS_DIR, cache_dir: Path | None = None) -> list[str]:
    """Unpack the committed models into the package's model cache; returns the names it wrote.

    A file models/<key>.txt.gz becomes <cache>/models/<key>.txt, where the package looks before
    training. <key> is the package's own cache key, so nothing here decides which model is used.
    Every model's sha256 must be listed in models/sha256; a file that does not match is an error,
    not a reason to retrain silently.
    """
    cache_dir = Path(cache_dir or os.environ.get("MTAH_CACHE_DIR") or PACKAGE_CACHE_DIR)
    listing = models_dir / "sha256"
    expected = {}
    if listing.is_file():
        expected = dict(line.split()[::-1] for line in listing.read_text().splitlines() if line)
    written = []
    for packed in sorted(models_dir.glob("*.txt.gz")):
        name = packed.name[: -len(".gz")]
        target = cache_dir / "models" / name
        if target.is_file():
            continue
        text = gzip.decompress(packed.read_bytes())
        if hashlib.sha256(text).hexdigest() != expected.get(name):
            raise ValueError(f"{packed}: sha256 does not match {listing}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(text)
        partial.replace(target)
        written.append(name)
    return written


@cache
def model(labels: str = "raw"):
    """(booster, feature function) of the package's final configuration, trained on `labels`.

    If models/ holds the model for exactly this configuration it is loaded. Otherwise the first
    call downloads the open data, degrades and featurises it and trains (minutes); later calls,
    in this process or another, load the cached model.
    """
    from possession import loader  # here, not at the top: loader reads the match data lazily

    degrade, _, submission, package_features = _package()
    if not os.environ.get("MTAH_NO_CACHE"):
        seed_model_cache()
    params = degrade.calibrated_params(container(loader.load_match(verbose=False)))
    feature_fn = package_features.FEATURE_SETS[submission.FINAL["features"]]
    train_cfg = {"weights": True} if submission.FINAL["weights"] else None
    booster = submission.train_final(
        feature_fn, label_function(labels), params, log=_log, train_cfg=train_cfg
    )
    return booster, feature_fn


def probabilities(match: Match, labels: str = "raw") -> np.ndarray:
    """(n, 3) float64: P(H), P(A), P(D) per frame from the supervised model, memoised on `match`."""

    def compute():
        booster, feature_fn = model(labels)
        _, _, submission, _ = _package()
        return np.asarray(submission.predict_match(booster, feature_fn, container(match)), float)

    return features._memo(match, ("supervised", labels), compute)
