"""Disk cache for the slow steps: degraded matches, feature matrices, fold predictions, models.

Everything lives in data/cache/ (gitignored and local: BAR-ATH-derived arrays never leave the
machine). A key hashes every input that changes the result -- the source code of the module that
computes it, parameters, seeds, and a fingerprint of the input match. Edit a feature block and only
that feature set is recomputed; nothing stale is ever reused.

    delete data/cache/            start clean
    set MTAH_NO_CACHE=1           bypass it (results are identical either way)

Feature matrices are stored as float16 on every path, cached or not, so a cache hit and a fresh
computation give byte-identical models -- the reproducibility rule cares about that.
"""

import hashlib
import inspect
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "data", "cache")
ENABLED = not os.environ.get("MTAH_NO_CACHE")
FEATURE_DTYPE = np.float16


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def digest(*parts):
    """Short stable hash of any JSON-able parts (None anywhere -> None: do not cache)."""
    if any(p is None for p in parts):
        return None
    blob = json.dumps(parts, sort_keys=True, default=_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def source(*objs):
    """Hash of the source code of functions / modules; None if it cannot be read."""
    h = hashlib.sha1()
    try:
        for o in objs:
            h.update(inspect.getsource(o).encode())
    except (OSError, TypeError):
        return None
    return h.hexdigest()[:20]


def fingerprint(m):
    """Cheap identity of a match container: name, length and a few checksums."""
    return [m["name"], int(len(m["label"])), int(m["px"].shape[1]),
            round(float(np.nansum(m["bx"])), 3), round(float(np.nansum(m["px"][::97])), 3),
            int(np.sum(m["label"].astype(np.int64)))]


def _path(kind, key, ext):
    folder = os.path.join(CACHE_DIR, kind)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, key + ext)


def array(kind, key, compute, mmap=True):
    """Array from the cache, or compute() it and store it. key None -> plain compute()."""
    if key is None or not ENABLED:
        return compute()
    path = _path(kind, key, ".npy")
    if not os.path.isfile(path):
        tmp = path[:-4] + ".tmp.npy"
        np.save(tmp, compute())
        os.replace(tmp, path)
    return np.load(path, mmap_mode="r" if mmap else None)


def value(kind, key, compute):
    """A JSON-able value from the cache, or compute() it and store it."""
    if key is None or not ENABLED:
        return compute()
    path = _path(kind, key, ".json")
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)
    out = compute()
    with open(path, "w") as handle:
        json.dump(out, handle, default=_default)
    return out


def container(kind, key, compute):
    """A match container from the cache, or compute() it and store it."""
    from solution import container as C
    if key is None or not ENABLED:
        return compute()
    path = _path(kind, key, ".npz")
    if os.path.isfile(path):
        return C.load(path)
    m = compute()
    C.save(m, path[:-4] + ".tmp.npz")
    os.replace(path[:-4] + ".tmp.npz", path)
    return m


def model(key, train):
    """A LightGBM booster from the cache, or train() it and store it."""
    import lightgbm as lgb
    if key is None or not ENABLED:
        return train()
    path = _path("models", key, ".txt")
    if os.path.isfile(path):
        return lgb.Booster(model_file=path)
    booster = train()
    booster.save_model(path[:-4] + ".tmp.txt")
    os.replace(path[:-4] + ".tmp.txt", path)
    return booster


def exists(kind, key, ext=".npy"):
    return key is not None and ENABLED and os.path.isfile(_path(kind, key, ext))
