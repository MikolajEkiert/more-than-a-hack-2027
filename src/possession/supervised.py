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
"""

import os
import sys
from functools import cache

import numpy as np

import possession_metrics as pm
from possession import features
from possession._paths import CACHE_DIR
from possession.match import Match

IDSSE_DIR = CACHE_DIR / "idsse"
PACKAGE_CACHE_DIR = CACHE_DIR / "solution"


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


@cache
def model(labels: str = "raw"):
    """(booster, feature function) of the package's final configuration, trained on `labels`.

    The first call downloads the open data, degrades and featurises it and trains (minutes);
    later calls, in this process or another, load the cached model.
    """
    from possession import loader  # here, not at the top: loader reads the match data lazily

    degrade, _, submission, package_features = _package()
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
