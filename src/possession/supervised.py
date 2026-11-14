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
PFF_DIR = CACHE_DIR / "pff"  # the PFF World Cup 2022 bundle, as unpacked (request form, not open)
PACKAGE_CACHE_DIR = CACHE_DIR / "solution"
MODELS_DIR = REPO_ROOT / "models"  # <cache key>.txt.gz, and `sha256` listing each model's hash


def _package():
    """The modules of solution/ this module uses, with their storage pointed at .cache/.

    The directories are read from the environment when the package is first imported, so they
    are set here, before the import, and nowhere else imports the package.
    """
    os.environ.setdefault("MTAH_IDSSE_DIR", str(IDSSE_DIR))
    os.environ.setdefault("MTAH_PFF_DIR", str(PFF_DIR))
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
def _setup():
    """(degradation parameters, feature function, training options) of the final configuration."""
    from possession import loader  # here, not at the top: loader reads the match data lazily

    degrade, _, submission, package_features = _package()
    params = degrade.calibrated_params(container(loader.load_match(verbose=False)))
    feature_fn = package_features.FEATURE_SETS[submission.FINAL["features"]]
    train_cfg = {"weights": True} if submission.FINAL["weights"] else None
    return params, feature_fn, train_cfg


DATA = ("idsse", "idsse+pff", "pff")  # what a model is trained on
PFF_ONLY_STRIDE = 8  # every 8th frame of 48 matches: about as many rows as the IDSSE model has


@cache
def model(labels: str = "raw", data: str = "idsse"):
    """(booster, feature function) of the package's final configuration, trained on `labels`.

    `data` names the training matches: "idsse" (the seven open matches, the submitted model),
    "idsse+pff" (plus the PFF training matches, in the package's own proportions: one
    orientation, one noise draw, every 15th frame) or "pff" (the PFF training matches alone).
    The PFF test matches (`pff_games`) are never trained on.

    If models/ holds the model for exactly this configuration it is loaded. Otherwise the first
    call downloads the open data, degrades and featurises it and trains (minutes); later calls,
    in this process or another, load the cached model.
    """
    if data not in DATA:
        raise ValueError(f"unknown training data {data!r}; known: {DATA}")
    _, _, submission, _ = _package()
    if not os.environ.get("MTAH_NO_CACHE"):
        seed_model_cache()
    params, feature_fn, train_cfg = _setup()
    label_fn = label_function(labels)
    if data == "pff":
        return _train_on_pff(feature_fn, label_fn, params, train_cfg), feature_fn
    extra = _pff_loaders(pff_games()[0]) if data == "idsse+pff" else None
    booster = submission.train_final(
        feature_fn, label_fn, params, log=_log, train_cfg=train_cfg, extra=extra
    )
    return booster, feature_fn


def _train_on_pff(feature_fn, label_fn, params, train_cfg):
    """One model on the PFF training matches alone, built from the package's own pieces."""
    from solution import cache as package_cache
    from solution import degrade
    from solution import evaluate_external as ev
    from solution import features as package_features
    from solution import labels as package_labels

    train_ids = pff_games()[0]
    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))

    def fit():
        prepared = _prepare_pff(train_ids, feature_fn, label_fn, params)
        x, y = ev.stack(prepared, stride=PFF_ONLY_STRIDE)
        weight = ev.stack_weights(prepared, stride=PFF_ONLY_STRIDE, cfg=cfg)
        return ev.train(x, y, weight=weight if cfg["weights"] else None)

    key = package_cache.digest(
        "pff-only",
        package_features.feature_key(feature_fn),
        package_labels.label_key(label_fn),
        sorted(dict(degrade.DEFAULTS, **params).items()),
        package_cache.source(degrade),
        list(train_ids),
        ev.LGB_PARAMS,
        ev.ROUNDS,
        PFF_ONLY_STRIDE,
        sorted(cfg.items()),
    )
    return package_cache.model(key, fit)


def _prepare_pff(ids, feature_fn, label_fn, params):
    """The package's `prepare` for PFF matches, exactly as `train_final` calls it for them."""
    from solution import evaluate_external as ev

    return ev.prepare(
        _pff_loaders(ids),
        feature_fn,
        label_fn,
        seed=0,
        degrade_params=params,
        log=lambda text: None,
        swap=False,
        light=True,
    )


def probabilities(match: Match, labels: str = "raw", data: str = "idsse") -> np.ndarray:
    """(n, 3) float64: P(H), P(A), P(D) per frame from the supervised model, memoised on `match`.

    A match the model could have been trained on is never scored by a model that saw it. Such a
    match carries a function under `HELD_OUT` that returns the prediction of a model trained
    without it (`idsse_conditions`, `pff_conditions`), and that is what is returned, whatever
    `labels` and `data` are asked for. A match named like a training match that has no such
    function is an error, not a silent leak.
    """

    def compute():
        held_out = match._cache.get(HELD_OUT)
        if held_out is not None:
            return np.asarray(held_out(labels, data), dtype=float)
        if match.game.startswith(TRAINING_PREFIXES):
            raise RuntimeError(
                f"{match.game} is named like a training match of the supervised model and has no "
                "held-out prediction attached; score it through supervised.idsse_conditions or "
                "supervised.pff_conditions"
            )
        return _predict(match, labels, data)

    return features._memo(match, ("supervised", labels, data), compute)


def _predict(match: Match, labels: str, data: str) -> np.ndarray:
    """Probabilities of the full model for `match`. The package caches the match's features on
    disk; with POSSESSION_NO_FEATURE_CACHE set they are computed and thrown away, which is the
    same numbers (the package rounds features to float16 on every path)."""
    booster, feature_fn = model(labels, data)
    _, _, submission, _ = _package()
    if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
        from solution import cache as package_cache

        x = feature_fn(container(match))[0].astype(package_cache.FEATURE_DTYPE)
        return np.asarray(booster.predict(x), float)
    return np.asarray(submission.predict_match(booster, feature_fn, container(match)), float)


# -- the training matches as a second validation set, each scored by a model that never saw it --

FPS = 25  # the package resamples every match to 25 frames per second
HELD_OUT = "supervised_held_out"  # Match._cache key: labels -> (n, 3) from a model without it
PACKAGE_LABELS = "supervised_package_labels"  # Match._cache key: labels -> the package's decode
TRAINING_PREFIXES = ("IDSSE-", "PFF-")  # how the package names the matches it is trained on
# Label readings the IDSSE matches are scored under. The first two have the possession density of
# the two Metrica readings (6.1 and 4.15 per live minute against 6.0 and 4.05) and are nobody's
# training labels; the provider's own flags ("raw", 8.3 per live minute) are reported only.
IDSSE_READINGS = ("controlled1.5", "controlled3.5", "raw")
IDSSE_RULE_READINGS = IDSSE_READINGS[:2]


def to_match(m: dict) -> Match:
    """A container of the package as a `Match`: the reverse of `container`.

    The container has no speed column; player speed is recomputed from the positions the same
    way as for the other external matches (`validation.speeds`).
    """
    from possession.validation import speeds

    xy = np.stack([m["px"], m["py"]], axis=2).astype(np.float64)
    n = len(xy)
    match = Match(
        game=str(m["name"]),
        fps=FPS,
        pitch=(float(m["pitch"][0]), float(m["pitch"][1])),
        ball=np.column_stack([m["bx"], m["by"], np.full(n, np.nan)]).astype(np.float64),
        players=np.concatenate([xy, speeds(xy, FPS)[..., None]], axis=2),
        team_of_track=np.asarray(m["team"], dtype=np.int8),
        half=np.asarray(m["half"], dtype=np.int8),
        tagging=None,
    )
    match.validate()
    return match


@cache
def _leave_one_out(labels: str, seed: int, data: str = "idsse") -> dict:
    """Every IDSSE match predicted by a model that was not trained on it.

    "idsse": a model trained on the other six (the package's harness). "idsse+pff": the other
    six plus the PFF training matches. "pff": the model trained on PFF alone, which has seen no
    IDSSE match at all. Returns {"probs": {name: (T, 3)}, "package": {name: codes}}; "package"
    is the package's own plain Viterbi with the penalties chosen on the other six matches, kept
    for comparison.
    """
    if data not in DATA:
        raise ValueError(f"unknown training data {data!r}; known: {DATA}")
    params, feature_fn, train_cfg = _setup()
    from solution import evaluate_external as ev
    from solution import idsse

    label_fn = label_function(labels)
    if data == "idsse":
        res = ev.run(
            feature_fn=feature_fn,
            label_fn=label_fn,
            seed=seed,
            degrade_params=params,
            bootstrap=0,
            log=_log,
            light=True,
            train_cfg=train_cfg,
        )
    else:
        prepared = ev.prepare(
            idsse.load_all(), feature_fn, label_fn, seed=seed, degrade_params=params, log=_log
        )
        if data == "pff":
            booster, _ = model(labels, "pff")
            probs = [booster.predict(d["X"]) for d in prepared]
        else:
            extra = _prepare_pff(pff_games()[0], feature_fn, label_fn, params)
            probs = ev.oof_with_extra(prepared, extra, log=_log, train_cfg=train_cfg)
        res = ev.score(probs, prepared, bootstrap=0)
    names = [d["m"]["name"] for d in res["data"]]
    return {
        "probs": dict(zip(names, res["probs"], strict=True)),
        "package": dict(zip(names, res["preds"], strict=True)),
    }


def idsse_conditions(seeds=(0, 1), readings=IDSSE_READINGS):
    """Yield (game, seed, match, {reading: labels}) for the seven IDSSE matches.

    The tracking is degraded exactly as for training (the package's degradation, calibrated on
    the competition tracking). The labels are derived from the provider's flags, never from the
    labels a model was trained on. Each match can only be scored out of fold: see `probabilities`.
    """
    params, feature_fn, _ = _setup()
    _, labels, _, _ = _package()
    from solution import evaluate_external as ev
    from solution import idsse

    letters = np.array(pm.LABELS)
    for seed in seeds:
        prepared = ev.prepare(
            idsse.load_all(),
            feature_fn,
            labels.raw,
            seed=seed,
            degrade_params=params,
            log=lambda text: None,
            swap=False,
        )
        for entry in prepared:
            m = entry["m"]  # degraded tracking with the provider's own labels
            name = str(m["name"])
            match = to_match(m)
            match._cache[HELD_OUT] = _from_fold("probs", name, seed)
            match._cache[PACKAGE_LABELS] = _from_fold("package", name, seed, letters)
            truths = {r: letters[np.asarray(label_function(r)(m))].astype("<U1") for r in readings}
            yield name.split("_deg")[0].lower(), seed, match, truths


def _from_fold(kind: str, name: str, seed: int, letters=None):
    """(labels, data) -> the out-of-fold result of `kind` for the match `name` under `seed`."""

    def get(labels: str = "raw", data: str = "idsse"):
        value = _leave_one_out(labels, seed, data)[kind][name]
        return value if letters is None else letters[np.asarray(value)].astype("<U1")

    return get


# -- PFF World Cup 2022: more training matches, and a test set with real tracking noise --------

PFF_READINGS = ("pff",)  # the provider's own labels: the owner changes at recorded on-ball events


def convert_pff(processes: int = 4) -> int:
    """Convert every PFF match under `PFF_DIR` to a container (the package's code, in parallel).

    Returns the number of converted matches. Raises if the bundle is not there: it is free but
    behind the provider's request form, so it cannot be downloaded by this code.
    """
    _package()
    from multiprocessing import Pool

    from solution import pff

    games = pff.find_games()
    if not games:
        raise FileNotFoundError(
            f"no PFF matches under {pff.RAW_DIR}; request the bundle at "
            "https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset and unpack it there"
        )
    os.makedirs(pff.NPZ_DIR, exist_ok=True)
    todo = [g for g in sorted(games) if not os.path.isfile(os.path.join(pff.NPZ_DIR, g + ".npz"))]
    if todo:
        with Pool(processes) as pool:
            pool.map(_convert_pff_game, todo, chunksize=1)
    return len(games)


def _convert_pff_game(game_id: str) -> None:
    _package()
    from solution import container as package_container
    from solution import pff

    match = pff.build_container(pff.find_games()[game_id])
    package_container.save(match, os.path.join(pff.NPZ_DIR, game_id + ".npz"))


@cache
def pff_games() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(training ids, test ids) of the PFF matches, fixed before anything was measured on them:
    sorted by id, every fourth match is a test match (16 of 64), the rest are for training."""
    _package()
    from solution import pff

    ids = sorted(path.stem for path in Path(pff.NPZ_DIR).glob("*.npz"))
    if not ids:
        raise FileNotFoundError(
            f"no converted PFF matches in {pff.NPZ_DIR}; run supervised.convert_pff() first"
        )
    test = tuple(ids[3::4])
    return tuple(g for g in ids if g not in test), test


def _pff_loaders(ids) -> list:
    """One zero-argument loader per match: 64 containers do not fit in memory at once."""
    _package()
    from solution import container as package_container
    from solution import pff

    paths = [os.path.join(pff.NPZ_DIR, g + ".npz") for g in ids]
    return [(lambda path=path: package_container.load(path)) for path in paths]


def pff_conditions(readings=PFF_READINGS):
    """Yield (game, seed, match, {reading: labels}) for the 16 PFF test matches.

    The tracking is used as delivered: it is broadcast tracking that already loses the ball on
    about four frames in ten, so nothing is simulated. No model is ever trained on these
    matches; the function attached to each match builds the prediction from the model asked for.
    """
    letters = np.array(pm.LABELS)
    for game_id, load in zip(pff_games()[1], _pff_loaders(pff_games()[1]), strict=True):
        m = load()
        match = to_match(m)
        match._cache[HELD_OUT] = _never_trained_on(match)
        truths = {r: letters[np.asarray(m["label"])].astype("<U1") for r in readings}
        yield f"pff-{game_id}", 0, match, truths


def _never_trained_on(match: Match):
    """(labels, data) -> probabilities for a PFF test match, which no model is trained on."""

    def get(labels: str = "raw", data: str = "idsse"):
        return _predict(match, labels, data)

    return get
