"""Frame-level ball possession for More than a Hack 2027.

The pipeline is three stages with a fixed contract (see docs/ARCHITECTURE.md):

    Match --ball_state--> p_alive --team--> p_home --decode--> labels (H/A/D/X)

Every algorithm is a *method* registered for one stage. A new idea is a new method, never a new
pipeline. `possession methods` lists what already exists.
"""

# ruff: noqa: I001  (order matters: _paths must run before anything imports organiser code)
from possession._paths import REPO_ROOT  # noqa: F401  (puts the repo root on sys.path first)
from possession import stages  # noqa: F401,E402  (importing registers every method)
