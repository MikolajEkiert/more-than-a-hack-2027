"""Training-label variants for the external data.

IDSSE's owner flag hands possession over about twice as often as the BAR-ATH public window
(8.4 vs 4.2 possessions per alive minute). Part is Barcelona's style, part may be definition:
CHALLENGE.md says a simple touch without control, or a deflection that does not intercept the
pass, is NOT a change of possession. merge_sandwiched() is that sentence as code -- a spell of the
other team shorter than tau, between two spells of the same team, is a touch, not a possession.

tau comes from the written definition, not from the public window (which itself keeps 0.8 s and
1.3 s sandwiched spells, so tau must stay small). Compare variants with the harness.
"""

import numpy as np

from solution import cache, container as C

pm = C.pm


def raw(m):
    return m["label"]


raw.cache_key = "raw"


def merge_sandwiched(codes, tau_s=0.5):
    """Absorb H/A spells shorter than tau_s that sit between two spells of the other team."""
    tau = int(round(tau_s * C.FPS))
    out = codes.copy()
    while True:
        segs = pm.segments(out, pm.valid_runs(out))
        changed = False
        for i in range(1, len(segs) - 1):
            a, b, k = segs[i]
            (pa, pb, pk), (na, nb, nk) = segs[i - 1], segs[i + 1]
            if (k in (C.H, C.A) and pk == nk and pk in (C.H, C.A) and pk != k
                    and b - a < tau and pb == a and b == na):
                out[a:b] = pk
                changed = True
        if not changed:
            return out


def harmonised(tau_s):
    """Label function for the harness: m -> codes with short sandwiched touches merged."""
    fn = lambda m: merge_sandwiched(m["label"], tau_s)  # noqa: E731
    fn.cache_key = "tau{}:{}".format(tau_s, cache.source(merge_sandwiched))
    return fn


def controlled_possession(codes, n_s=2.0):
    """A hand-over counts only if the new team keeps the ball n_s seconds (2-3 s in several
    competitor repos; CHALLENGE.md's own proxy: "a player keeps the ball for a few seconds").
    A shorter spell right after the other team's spell is given back to that team; spells right
    after a stoppage are restarts and always stay."""
    n = int(round(n_s * C.FPS))
    out = codes.copy()
    while True:
        segs = pm.segments(out, pm.valid_runs(out))
        changed = False
        for i in range(1, len(segs)):
            a, b, k = segs[i]
            pa, pb, pk = segs[i - 1]
            if k in (C.H, C.A) and pk in (C.H, C.A) and pk != k and b - a < n and pb == a:
                out[a:b] = pk
                changed = True
                break           # segments shift after a merge; recompute
        if not changed:
            return out


def controlled(n_s):
    fn = lambda m: controlled_possession(m["label"], n_s)  # noqa: E731
    fn.cache_key = "controlled{}:{}".format(n_s, cache.source(controlled_possession))
    return fn


def label_key(fn):
    return getattr(fn, "cache_key", None) or cache.source(fn)
