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
from possession import features as derived
from possession import sequence, team_features
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
def model(labels: str = "raw", data: str = "idsse", regime: str = "none", features: str = "v1"):
    """(booster, feature function) of the package's final configuration, trained on `labels`.

    `data` names the training matches: "idsse" (the seven open matches, the submitted model),
    "idsse+pff" (plus the PFF training matches, in the package's own proportions: one
    orientation, one noise draw, every 15th frame) or "pff" (the PFF training matches alone).
    The PFF test matches (`pff_games`) are never trained on. `regime` is the ball regime of the
    training matches (`possession.regime`): "none" is the package's own degradation, the others
    lose the ball as automatic tracking does (IDSSE only). `features` is the feature set
    (`FEATURE_SETS`): "v1" is the package's own; the others append the block of
    `possession.team_features` to it (task 14) and are trained by `_train_final_team`.

    If models/ holds the model for exactly this configuration it is loaded. Otherwise the first
    call downloads the open data, degrades and featurises it and trains (minutes); later calls,
    in this process or another, load the cached model.
    """
    if features != "v1":
        _check_team(features, data, regime)
        return _train_final_team(labels, features, data), _setup()[1]
    if data not in DATA:
        raise ValueError(f"unknown training data {data!r}; known: {DATA}")
    _, _, submission, _ = _package()
    if not os.environ.get("MTAH_NO_CACHE"):
        seed_model_cache()
    params, feature_fn, train_cfg = _setup()
    label_fn = label_function(labels)
    if regime != "none":
        if data != "idsse":
            raise ValueError(f"regime {regime!r} is only defined for data='idsse'")
        return _train_final_regime(feature_fn, label_fn, params, train_cfg, regime), feature_fn
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


def _train_final_regime(feature_fn, label_fn, params, train_cfg, regime_name):
    """The package's `train_final` (its seed loop, unchanged) on IDSSE matches whose ball follows
    `regime_name` (`possession.regime`), cached in the package's cache under its own key."""
    from possession import regime as regime_mod
    from solution import cache as package_cache
    from solution import degrade, idsse
    from solution import evaluate_external as ev
    from solution import features as package_features
    from solution import labels as package_labels

    _, _, submission, _ = _package()
    seeds = tuple(submission.DEGRADE_SEEDS)
    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))

    def fit():
        raws = idsse.load_all()
        xs, ys, ws = [], [], []
        for seed in seeds:
            data = ev.prepare(
                regime_mod.regime_inputs(raws, regime_name, seed),
                feature_fn,
                label_fn,
                seed=seed,
                degrade_params=regime_mod.regime_params(params, regime_name, seed),
                log=_log,
            )
            x, y = ev.stack(data)
            xs.append(x)
            ys.append(y)
            ws.append(ev.stack_weights(data, cfg=cfg))
            del data
        weight = np.concatenate(ws) if cfg["weights"] else None
        return ev.train(np.concatenate(xs), np.concatenate(ys), weight=weight)

    key = package_cache.digest(
        "final-regime",
        regime_name,
        regime_mod.regime_key(regime_name),
        package_features.feature_key(feature_fn),
        package_labels.label_key(label_fn),
        sorted(dict(degrade.DEFAULTS, **params).items()),
        list(seeds),
        package_cache.source(degrade),
        sorted(idsse.MATCHES),
        ev.LGB_PARAMS,
        ev.ROUNDS,
        ev.TRAIN_STRIDE,
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


def probabilities(
    match: Match,
    labels: str = "raw",
    data: str = "idsse",
    regime: str = "none",
    tta: bool = False,
    features: str = "v1",
) -> np.ndarray:
    """(n, 3) float64: P(H), P(A), P(D) per frame from the supervised model, memoised on `match`.

    `regime` picks the model trained in that ball regime (`model`); `tta` averages the prediction
    of the match with that of the same match with home and away exchanged (columns swapped back);
    `features` picks the feature set (`FEATURE_SETS`; "v1" is the submitted model's).

    A match the model could have been trained on is never scored by a model that saw it. Such a
    match carries a function under `HELD_OUT` that returns the prediction of a model trained
    without it (`idsse_conditions`, `pff_conditions`), and that is what is returned, whatever
    `labels` and `data` are asked for. A match named like a training match that has no such
    function is an error, not a silent leak.
    """

    def compute():
        held_out = match._cache.get(HELD_OUT)
        if held_out is not None:
            return np.asarray(
                held_out(labels, data, regime=regime, tta=tta, features=features), dtype=float
            )
        if match.game.startswith(TRAINING_PREFIXES):
            raise RuntimeError(
                f"{match.game} is named like a training match of the supervised model and has no "
                "held-out prediction attached; score it through supervised.idsse_conditions or "
                "supervised.pff_conditions"
            )
        return _predict(match, labels, data, regime, tta, features)

    key = ("supervised", labels, data, regime, tta, features)
    return derived._memo(match, key, compute)


def features_of(match: Match) -> np.ndarray:
    """(T, K) float16: the package's per-frame features of `match` (the final feature set), from
    the same disk cache as `_predict` (the temporal network of `sequence.py` reads them)."""
    _, feature_fn, _ = _setup()
    _package()
    from solution import cache as package_cache
    from solution import features as package_features

    m = container(match)
    if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
        return feature_fn(m)[0].astype(package_cache.FEATURE_DTYPE)
    key = package_cache.digest("match", package_cache.fingerprint(m))
    return package_features.cached(feature_fn, m, "orig", key)[0]


def training_weights(label: np.ndarray) -> np.ndarray:
    """The package's per-frame sample weights of the final configuration (boundaries and short
    possessions count more) for a label array in its codes."""
    _, _, train_cfg = _setup()
    from solution import evaluate_external as ev

    return ev.sample_weights(np.asarray(label), dict(ev.DEFAULT_CFG, **(train_cfg or {})))


def idsse_prepared(labels: str, seed: int, part: str, swap: bool = True) -> list:
    """The package's prepared IDSSE matches (features of both orientations, labels) of noise seed
    `seed` in the single ball regime `part` (`possession.regime`)."""
    from possession import regime as regime_mod
    from solution import evaluate_external as ev
    from solution import idsse

    params, feature_fn, _ = _setup()
    return ev.prepare(
        regime_mod.regime_inputs(idsse.load_all(), part, seed),
        feature_fn,
        label_function(labels),
        seed=seed,
        degrade_params=regime_mod.regime_params(params, part, seed),
        log=_log,
        swap=swap,
    )


SWAP_COLUMNS = [1, 0, 2]  # P(H), P(A), P(D) of the swapped view back in the original's order


def _predict(
    match: Match,
    labels: str,
    data: str,
    regime: str = "none",
    tta: bool = False,
    features: str = "v1",
) -> np.ndarray:
    """Probabilities of the full model for `match`. The package caches the match's features on
    disk; with POSSESSION_NO_FEATURE_CACHE set they are computed and thrown away, which is the
    same numbers (the package rounds features to float16 on every path)."""
    if features != "v1":
        return _predict_team(match, labels, data, regime, tta, features)
    booster, feature_fn = model(labels, data, regime)
    _, _, submission, _ = _package()
    from solution import cache as package_cache
    from solution import container as package_container
    from solution import features as package_features

    m = container(match)
    views = [(m, "orig")] + ([(package_container.swap_teams(m), "swap")] if tta else [])
    out = []
    for view, tag in views:
        if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
            x = feature_fn(view)[0].astype(package_cache.FEATURE_DTYPE)
            p = np.asarray(booster.predict(x), float)
        elif tag == "orig":
            p = np.asarray(submission.predict_match(booster, feature_fn, view), float)
        else:
            key = package_cache.digest("match", package_cache.fingerprint(m))
            x, _ = package_features.cached(feature_fn, view, tag, key)
            p = np.asarray(booster.predict(x), float)
        out.append(p if tag == "orig" else p[:, SWAP_COLUMNS])
    return np.mean(out, axis=0) if len(out) > 1 else out[0]


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


# The IDSSE matches as validation inputs: "plain" is the package's degradation (suite "idsse"),
# "auto" loses the ball as automatic tracking does, with the profile fitted to the competition
# match (suite "idsse_auto"; docs/FUSION.md section 10).
INPUT_REGIMES = {"plain": "none", "auto": "auto_ml"}


def _idsse_inputs(raws, input_regime: str, seed: int, feature_fn, label_fn, params, swap=True):
    """The package's `prepare` of the IDSSE matches under one validation input regime."""
    from possession import regime as regime_mod
    from solution import evaluate_external as ev

    name = INPUT_REGIMES[input_regime]
    return ev.prepare(
        regime_mod.regime_inputs(raws, name, seed),
        feature_fn,
        label_fn,
        seed=seed,
        degrade_params=regime_mod.regime_params(params, name, seed),
        log=_log,
        swap=swap,
    )


@cache
def _regime_folds(labels: str, seed: int, regime: str) -> dict:
    """{(match name, input regime, view): (T, 3)} for every IDSSE match, predicted by a model
    trained on the other six (noise seed `seed`) in the ball regime `regime`, in both validation
    input regimes and both views ("orig", and "swap" with home and away exchanged).

    Like the package's harness, a fold uses the copies of one noise seed. A single regime trains
    on that seed's copies in that regime, every `TRAIN_STRIDE`-th frame: with regime "none" this is
    exactly the package's harness (`ev.run` keeps no fold models, so they are retrained here with
    the same `ev.fit` on the same prepared data, and checked bit for bit against `_leave_one_out`).
    A mixed regime (`regime.MIXED`, half the final model's seeds in each regime) trains on that
    seed's copies in each of its regimes, every 2 x `TRAIN_STRIDE`-th frame of each: the same
    number of rows as a single-regime fold, in the final model's proportions.
    """
    from possession import regime as regime_mod
    from solution import cache as package_cache
    from solution import evaluate_external as ev
    from solution import idsse

    params, feature_fn, train_cfg = _setup()
    label_fn = label_function(labels)
    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))
    raws = idsse.load_all()
    parts = regime_mod.MIXED.get(regime, (regime,))
    stride = ev.TRAIN_STRIDE * len(parts)
    prepared = {}
    for part in parts:
        prepared[part] = ev.prepare(
            regime_mod.regime_inputs(raws, part, seed),
            feature_fn,
            label_fn,
            seed=seed,
            degrade_params=regime_mod.regime_params(params, part, seed),
            log=_log,
        )
    inputs = {
        name: prepared[INPUT_REGIMES[name]]
        if INPUT_REGIMES[name] in prepared
        else _idsse_inputs(raws, name, seed, feature_fn, label_fn, params)
        for name in INPUT_REGIMES
    }
    run_key = package_cache.digest(
        "fold-regime",
        regime_mod.regime_key(regime),
        [[d["key"] for d in prepared[part]] for part in parts],
        ev.LGB_PARAMS,
        ev.ROUNDS,
        stride,
        sorted(cfg.items()),
    )

    def fit(k):
        if len(parts) == 1:
            return ev.fit(prepared[parts[0]], skip=k, cfg=cfg)
        xs, ys, ws = [], [], []
        for part in parts:
            x, y = ev.stack(prepared[part], skip=k, stride=stride)
            xs.append(x)
            ys.append(y)
            ws.append(ev.stack_weights(prepared[part], skip=k, stride=stride, cfg=cfg))
        weight = np.concatenate(ws) if cfg["weights"] else None
        return ev.train(np.concatenate(xs), np.concatenate(ys), weight=weight)

    out = {}
    for k in range(len(raws)):
        wanted = [(inp, view) for inp in INPUT_REGIMES for view in ("orig", "swap")]
        keys = {
            (inp, view): package_cache.digest(run_key, k, inputs[inp][k]["key"], view)
            for inp, view in wanted
        }
        booster = None
        if not all(package_cache.exists("folds", key) for key in keys.values()):
            booster = fit(k)
            _log(f"fold {k} of regime {regime} (seed {seed}) trained")
        for (inp, view), key in keys.items():
            entry = inputs[inp][k]
            x = entry["X"] if view == "orig" else entry["Xs"]
            out[(str(entry["m"]["name"]), inp, view)] = package_cache.array(
                "folds", key, lambda x=x, b=booster: b.predict(x), mmap=False
            )
    if regime == "none":  # the same folds as the package's harness, bit for bit, or nothing
        reference = _leave_one_out(labels, seed)["probs"]
        for name, probs in reference.items():
            if not np.array_equal(np.asarray(probs), np.asarray(out[(name, "plain", "orig")])):
                raise RuntimeError(f"fold loop differs from the package's harness on {name}")
    return out


def idsse_conditions(seeds=(0, 1), readings=IDSSE_READINGS, input_regime: str = "plain"):
    """Yield (game, seed, match, {reading: labels}) for the seven IDSSE matches.

    The tracking is degraded as for training: the package's degradation, calibrated on the
    competition tracking ("plain"), or the automatic-tracking ball regime of the profile fitted to
    the competition match ("auto"; `INPUT_REGIMES`). The labels are derived from the provider's
    flags, never from the labels a model was trained on. Each match can only be scored out of
    fold: see `probabilities`.
    """
    if input_regime not in INPUT_REGIMES:
        raise ValueError(f"unknown input regime {input_regime!r}; known: {sorted(INPUT_REGIMES)}")
    params, feature_fn, _ = _setup()
    _, labels, _, _ = _package()
    from solution import idsse

    letters = np.array(pm.LABELS)
    raws = idsse.load_all()
    for seed in seeds:
        prepared = _idsse_inputs(
            raws, input_regime, seed, feature_fn, labels.raw, params, swap=False
        )
        for entry in prepared:
            m = entry["m"]  # degraded tracking with the provider's own labels
            name = str(m["name"])
            match = to_match(m)
            match._cache[HELD_OUT] = _from_fold("probs", name, seed, input_regime=input_regime)
            match._cache[sequence.HELD_OUT] = sequence.held_out(name, input_regime)
            if input_regime == "plain":
                match._cache[PACKAGE_LABELS] = _from_fold("package", name, seed, letters)
            truths = {r: letters[np.asarray(label_function(r)(m))].astype("<U1") for r in readings}
            yield name.split("_deg")[0].lower(), seed, match, truths


def _from_fold(kind: str, name: str, seed: int, letters=None, input_regime: str = "plain"):
    """(labels, data, regime, tta, features) -> the out-of-fold result of `kind` for the match
    `name`."""

    def get(
        labels: str = "raw",
        data: str = "idsse",
        regime: str = "none",
        tta: bool = False,
        features: str = "v1",
    ):
        if features != "v1":
            if kind != "probs":
                raise ValueError(f"no out-of-fold {kind!r} for the feature set {features!r}")
            _check_team(features, data, regime, tta)
            return _team_folds(labels, seed, features, data)[(name, input_regime)]
        if kind == "package" or (input_regime == "plain" and regime == "none" and not tta):
            value = _leave_one_out(labels, seed, data)[kind][name]
            return value if letters is None else letters[np.asarray(value)].astype("<U1")
        if data != "idsse":
            raise ValueError("ball regimes and test-time augmentation are only for data='idsse'")
        folds = _regime_folds(labels, seed, regime)
        p = np.asarray(folds[(name, input_regime, "orig")], float)
        if tta:
            p = (p + np.asarray(folds[(name, input_regime, "swap")], float)[:, SWAP_COLUMNS]) / 2
        return p

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
        match._cache[sequence.HELD_OUT] = sequence.never_trained_on(match)
        truths = {r: letters[np.asarray(m["label"])].astype("<U1") for r in readings}
        yield f"pff-{game_id}", 0, match, truths


def _never_trained_on(match: Match):
    """(labels, data, ...) -> probabilities for a PFF test match, which no model is trained on."""

    def get(
        labels: str = "raw",
        data: str = "idsse",
        regime: str = "none",
        tta: bool = False,
        features: str = "v1",
    ):
        return _predict(match, labels, data, regime, tta, features)

    return get


# -- an extra feature block (task 14): the package's features plus possession.team_features ----

# Feature sets of the model. "v1" is the package's own, the submitted model, and nothing below is
# on its path. The others append the block of `possession.team_features` to every row and differ
# only in the number of boosting rounds (None: the package's ROUNDS). They are trained here
# exactly as the package trains "v1" (the same matches, degradation, labels, noise seeds,
# orientations, stride, sample weights and LightGBM parameters) and cached under TEAM_DIR, never
# in the package's cache.
FEATURE_SETS = {"v1": None, "v1+team": {"rounds": None}, "v1+team-r600": {"rounds": 600}}
TEAM_DIR = CACHE_DIR / "team_features"  # blocks/, models/, folds/
TEAM_MODELS_DIR = MODELS_DIR / "team"  # committed final models of the feature sets: <key>.txt.gz


def seed_team_models(models_dir: Path = TEAM_MODELS_DIR, cache_dir: Path | None = None) -> list:
    """Unpack the committed final models of the team-feature sets into TEAM_DIR/models.

    The same mechanism as `seed_model_cache` for the package's model: a file is named by its
    cache key, so it is used only if it is exactly the model this code would train, and a
    sha256 that does not match `models/team/sha256` is an error. Returns the names written.
    """
    return seed_model_cache(models_dir, cache_dir or TEAM_DIR)


# Training data of the team-feature models (task 17): the seven IDSSE matches (version 4's model,
# whose cache key is unchanged) and, appended to them, the extra matches of the providers named
# after the "+". The extra matches enter as the package adds PFF to its own model (`train_final`
# with `extra`): one noise draw, one orientation, every `EXTRA_STRIDE`-th graded frame.
TEAM_DATA = {
    "idsse": (),
    "idsse+pff": ("pff",),  # the 48 PFF training matches (`pff_games`), never the 16 test ones
    "idsse+sc": ("sc",),  # the 20 SkillCorner open matches (`possession.skillcorner`)
    "idsse+pff+sc": ("pff", "sc"),
}
EXTRA_STRIDE = 15  # the package's PFF_STRIDE


def _check_team(features: str, data: str = "idsse", regime: str = "none", tta: bool = False):
    """Raise unless `features` names a feature set that exists for these options."""
    if features not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {features!r}; known: {sorted(FEATURE_SETS)}")
    if features != "v1" and (data not in TEAM_DATA or regime != "none" or tta):
        raise ValueError(
            f"the feature set {features!r} exists for data in {sorted(TEAM_DATA)}, "
            "regime='none' only"
        )


def _cache_on() -> bool:
    return not os.environ.get("MTAH_NO_CACHE")  # the package's switch, honoured here too


def _cached_npy(path: Path, compute, mmap: bool = True) -> np.ndarray:
    """An array from `path`, or compute() it and store it there (atomically)."""
    if not _cache_on():
        return compute()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(partial, compute())
        partial.replace(path)
    return np.load(path, mmap_mode="r" if mmap else None)


def _cached_booster(path: Path, fit):
    """A LightGBM booster from `path`, or fit() it and store it there. It is always read back from
    the file, so the first call and every later one predict with the same model text."""
    import lightgbm as lgb

    if not _cache_on():
        return fit()
    if not path.is_file():
        booster = fit()
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp.txt")
        booster.save_model(str(partial))
        partial.replace(path)
    return lgb.Booster(model_file=str(path))


def _content_key(m: dict) -> str:
    """Hash of every array of container `m` that the block reads."""
    h = hashlib.sha256()
    for name in ("px", "py", "bx", "by", "team", "half", "pitch"):
        a = np.ascontiguousarray(m[name])
        h.update(f"{name}:{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:24]


def team_block(m: dict, view: str = "orig") -> np.ndarray:
    """(T, K) float16: the block of `possession.team_features` for container `m`, or for `m` with
    home and away exchanged (view "swap", the package's second orientation). Cached under
    TEAM_DIR/blocks by the block's key and the container's content; POSSESSION_NO_FEATURE_CACHE
    computes it without the cache (the same numbers: float16 on every path)."""
    if view not in ("orig", "swap"):
        raise ValueError(f"unknown view {view!r}")
    source = team_features.swapped(m) if view == "swap" else m

    def compute():
        return team_features.build(source)[0]

    if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
        return compute()
    key = hashlib.sha256(
        f"{team_features.block_key()}:{_content_key(source)}:{view}".encode()
    ).hexdigest()[:24]
    return _cached_npy(TEAM_DIR / "blocks" / f"{key}.npy", compute)


def _team_blocks(prepared: list) -> list[dict]:
    """The block of each prepared match, in both orientations when the match has both."""
    return [
        {
            "orig": team_block(d["m"]),
            "swap": team_block(d["m"], "swap") if d.get("Xs") is not None else None,
        }
        for d in prepared
    ]


def _team_stack(prepared: list, blocks: list, skip=None) -> tuple[np.ndarray, np.ndarray]:
    """`ev.stack(prepared, skip)` with each match's block appended to its rows: the same rows in
    the same order (every TRAIN_STRIDE-th graded frame, the original orientation, then the
    swapped one), float16 as the package's matrix."""
    from solution import container as package_container
    from solution import evaluate_external as ev

    skip = set() if skip is None else ({skip} if isinstance(skip, int | np.integer) else set(skip))
    xs, ys = [], []
    for j, d in enumerate(prepared):
        if j in skip:
            continue
        views = [(d["X"], blocks[j]["orig"], d["m"]["label"])]
        if d.get("Xs") is not None:
            views.append((d["Xs"], blocks[j]["swap"], d["ys"]))
        for x, extra, y in views:
            rows = np.flatnonzero(y != package_container.X)[:: ev.TRAIN_STRIDE]
            xs.append(np.hstack([x[rows], extra[rows]]))
            ys.append(y[rows])
    return np.concatenate(xs), np.concatenate(ys)


def _team_rows(prepared: list, cfg: dict, skip=None) -> tuple:
    """(x, y, sample weights) of the prepared matches of one noise seed except `skip`, the block
    appended: what `ev.fit` and `train_final` stack for that seed."""
    from solution import evaluate_external as ev

    x, y = _team_stack(prepared, _team_blocks(prepared), skip=skip)
    return x, y, ev.stack_weights(prepared, skip, ev.TRAIN_STRIDE, cfg)


def _fit_team(parts: list, cfg: dict, rounds: int):
    """LightGBM on the rows of every part (one per noise seed, in order), with the package's
    parameters and, if the configuration asks for them, its sample weights."""
    from solution import evaluate_external as ev

    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    weight = np.concatenate([p[2] for p in parts]) if cfg["weights"] else None
    parts.clear()  # the caller's rows are now in x: free them before LightGBM copies x
    return ev.train(x, y, weight=weight, rounds=rounds)


def _team_rounds(features: str) -> int:
    from solution import evaluate_external as ev

    return FEATURE_SETS[features]["rounds"] or ev.ROUNDS


def _team_key(labels: str, features: str) -> list:
    """Everything a model of the feature set `features` depends on, beside its matches."""
    import inspect

    params, feature_fn, train_cfg = _setup()
    degrade, _, _, package_features = _package()
    from solution import cache as package_cache
    from solution import evaluate_external as ev
    from solution import labels as package_labels

    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))
    code = "".join(
        inspect.getsource(fn) for fn in (_team_stack, _team_blocks, _team_rows, _fit_team)
    )
    return [
        features,
        team_features.block_key(),
        hashlib.sha256(code.encode()).hexdigest()[:16],
        package_features.feature_key(feature_fn),
        package_labels.label_key(label_function(labels)),
        sorted(dict(degrade.DEFAULTS, **params).items()),
        package_cache.source(degrade),
        ev.LGB_PARAMS,
        _team_rounds(features),
        ev.TRAIN_STRIDE,
        sorted(cfg.items()),
    ]


@cache
def _train_final_team(labels: str, features: str, data: str = "idsse"):
    """The package's `train_final` with the block appended to every row: all seven IDSSE matches,
    noise seeds `DEGRADE_SEEDS`, both orientations, every TRAIN_STRIDE-th graded frame, the
    package's sample weights, LightGBM parameters and (unless the set says otherwise) rounds.
    With `data` other than "idsse" the rows of the extra matches of `TEAM_DATA[data]` are
    appended (`_extra_rows`). Cached under TEAM_DIR/models."""
    params, feature_fn, train_cfg = _setup()
    _, _, submission, _ = _package()
    from solution import cache as package_cache
    from solution import evaluate_external as ev
    from solution import idsse

    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))
    seeds = tuple(submission.DEGRADE_SEEDS)
    label_fn = label_function(labels)

    def fit():
        parts = []
        for seed in seeds:  # one seed's matches in memory at a time, as in `train_final`
            prepared = ev.prepare(
                idsse.load_all(), feature_fn, label_fn, seed=seed, degrade_params=params, log=_log
            )
            parts.append(_team_rows(prepared, cfg))
            del prepared
        if TEAM_DATA[data]:
            parts.append(_extra_rows(data, labels, cfg))
        return _fit_team(parts, cfg, _team_rounds(features))

    key = package_cache.digest(
        "team-final", *_team_key(labels, features), list(seeds), sorted(idsse.MATCHES)
    )
    if TEAM_DATA[data]:  # version 4's key (data "idsse") stays as it was
        key = package_cache.digest(key, *_extra_key(data))
    if _cache_on():
        seed_team_models()
    return _cached_booster(TEAM_DIR / "models" / f"{key}.txt", fit)


@cache
def _team_folds(labels: str, seed: int, features: str, data: str = "idsse") -> dict:
    """{(match name, input regime): (T, 3)} for every IDSSE match of noise seed `seed`, predicted
    by a model of the feature set `features` trained on the other six, as the package's harness
    does it (`ev.run`: that seed's copies, both orientations, every TRAIN_STRIDE-th graded frame,
    sample weights), in both validation input regimes (`INPUT_REGIMES`). With `data` other than
    "idsse" every fold also trains on all the extra matches (`_extra_rows`), as the package's
    `oof_with_extra` does; the held-out IDSSE match is never in. The predictions are cached under
    TEAM_DIR/folds; the fold models are not kept."""
    params, feature_fn, train_cfg = _setup()
    from solution import cache as package_cache
    from solution import evaluate_external as ev
    from solution import idsse

    cfg = dict(ev.DEFAULT_CFG, **(train_cfg or {}))
    label_fn = label_function(labels)
    raws = idsse.load_all()
    prepared = ev.prepare(raws, feature_fn, label_fn, seed=seed, degrade_params=params, log=_log)
    inputs = {"plain": prepared}
    inputs["auto"] = _idsse_inputs(raws, "auto", seed, feature_fn, label_fn, params, swap=False)
    run_key = package_cache.digest(
        "team-fold", *_team_key(labels, features), seed, [d["key"] for d in prepared]
    )
    if TEAM_DATA[data]:  # version 4's folds (data "idsse") keep their key
        run_key = package_cache.digest(run_key, *_extra_key(data))
    out = {}
    for k in range(len(prepared)):
        paths = {
            inp: TEAM_DIR
            / "folds"
            / f"{package_cache.digest(run_key, k, inputs[inp][k]['key'])}.npy"
            for inp in INPUT_REGIMES
        }
        booster = None
        if not (_cache_on() and all(path.is_file() for path in paths.values())):
            parts = [_team_rows(prepared, cfg, skip=k)]
            if TEAM_DATA[data]:
                parts.append(_extra_rows(data, labels, cfg))
            booster = _fit_team(parts, cfg, _team_rounds(features))
            _log(f"team fold {k} of seed {seed} ({features}, {data}) trained")
        for inp, path in paths.items():
            entry = inputs[inp][k]

            def predict(entry=entry, booster=booster):
                x = np.hstack([entry["X"], team_block(entry["m"])])
                return np.asarray(booster.predict(x), dtype=float)

            out[(str(entry["m"]["name"]), inp)] = _cached_npy(path, predict, mmap=False)
    return out


def _predict_team(
    match: Match, labels: str, data: str, regime: str, tta: bool, features: str
) -> np.ndarray:
    """Probabilities of the final model of the feature set `features` for `match`: the package's
    features through its own cache (as `_predict` reads them), the block appended."""
    _check_team(features, data, regime, tta)
    booster, feature_fn = model(labels, data, regime, features)
    _package()
    from solution import cache as package_cache
    from solution import features as package_features

    m = container(match)
    if os.environ.get("POSSESSION_NO_FEATURE_CACHE"):
        x = feature_fn(m)[0].astype(package_cache.FEATURE_DTYPE)
    else:
        key = package_cache.digest("match", package_cache.fingerprint(m))
        x, _ = package_features.cached(feature_fn, m, "orig", key)
    return np.asarray(booster.predict(np.hstack([x, team_block(m)])), dtype=float)


# -- extra training matches of other providers for the team-feature models (task 17) -----------


def _extra_sources(data: str) -> list[tuple[str, list]]:
    """[(provider, [(match id, zero-argument loader of its container), ...])] of `data`'s extra
    matches, in a fixed order. PFF: the 48 training matches of `pff_games`, never its 16 test
    matches."""
    out = []
    for provider in TEAM_DATA[data]:
        if provider == "pff":
            ids = pff_games()[0]
            out.append((provider, list(zip(ids, _pff_loaders(ids), strict=True))))
        elif provider == "sc":
            from possession import skillcorner

            ids = skillcorner.match_ids()
            out.append((provider, list(zip(ids, skillcorner.loaders(), strict=True))))
        else:
            raise ValueError(f"unknown extra provider {provider!r}")
    return out


@cache
def _extra_key(data: str) -> tuple:
    """What the extra rows of `data` depend on beyond `_team_key`: the providers, the content of
    every extra match (positions, teams, halves, pitch and labels; not the container file's bytes,
    which carry a time stamp), the stride and this code."""
    import inspect

    contents = []
    for provider, matches in _extra_sources(data):
        for game, load in matches:
            m = load()
            label = hashlib.sha256(np.ascontiguousarray(m["label"]).tobytes()).hexdigest()[:24]
            contents.append((provider, game, _content_key(m), label))
    code = "".join(inspect.getsource(fn) for fn in (_extra_match_rows, _extra_rows_cached))
    return ("extra", data, contents, EXTRA_STRIDE, hashlib.sha256(code.encode()).hexdigest()[:16])


def _extra_match_rows(raw: dict, index: int, label_fn, cfg: dict) -> tuple:
    """(x, y, w) of one extra match, the block appended: what the package's `prepare` (noise seed
    DEGRADE_SEEDS[0], the match at position `index`, one orientation, the ball loss topped up for
    what the provider already lacks) and `stack` / `stack_weights` with stride EXTRA_STRIDE make
    of it, as `train_final` does for its `extra`. One match at a time: 48 containers do not fit in
    memory together."""
    params, feature_fn, _ = _setup()
    degrade, _, submission, package_features = _package()
    from solution import cache as package_cache
    from solution import container as package_container
    from solution import evaluate_external as ev
    from solution import labels as package_labels

    dseed = submission.DEGRADE_SEEDS[0] * 100 + index  # how `ev.prepare` numbers its matches
    params_i = ev._top_up(raw, params)
    mkey = degrade.degrade_key(raw, dseed, params_i)
    m = package_cache.container(
        "degraded", mkey, lambda: degrade.degrade(raw, seed=dseed, **(params_i or {}))
    )
    lab = package_cache.array(
        "labels",
        package_cache.digest(package_labels.label_key(label_fn), package_cache.fingerprint(raw)),
        lambda: label_fn(raw),
        mmap=False,
    )
    m = dict(m, label=np.asarray(lab, np.int8))
    x, _ = package_features.cached(feature_fn, m, "orig", mkey)
    rows = np.flatnonzero(m["label"] != package_container.X)[::EXTRA_STRIDE]
    w = ev.sample_weights(m["label"], cfg)[rows]
    return np.hstack([x[rows], team_block(m)[rows]]), m["label"][rows], w


def _extra_rows(data: str, labels: str, cfg: dict) -> tuple:
    """(x, y, w) of every extra match of `data`, in order, memoised in the process: every fold and
    the final model append the same rows."""
    return _extra_rows_cached(data, labels, tuple(sorted(cfg.items())))


@cache
def _extra_rows_cached(data: str, labels: str, cfg_items: tuple) -> tuple:
    cfg = dict(cfg_items)
    label_fn = label_function(labels)
    xs, ys, ws = [], [], []
    for _, matches in _extra_sources(data):
        for index, (game, load) in enumerate(matches):
            x, y, w = _extra_match_rows(load(), index, label_fn, cfg)
            xs.append(x)
            ys.append(y)
            ws.append(w)
            _log(f"extra rows {game}: {len(y)}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
