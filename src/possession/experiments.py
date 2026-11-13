"""The experiment log: one JSON file per run in experiments/runs/, committed.

One file per run, never a shared file that everyone appends to, so three people logging runs in
parallel never produce a merge conflict. `possession log` renders them as a table.

What a run records is aggregates only: config, code version, diagnostics and the public sanity
scores. Never frame-level data.
"""

import getpass
import json
import subprocess
from datetime import UTC, datetime

from possession._paths import REPO_ROOT, RUNS_DIR


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def record(
    config,
    sha256: str,
    diagnostics: dict,
    sanity: dict | None,
    seconds: float,
    note="",
    external: dict | None = None,
):
    """Write one run JSON. `external` holds a score on another match (possession validate)."""
    now = datetime.now(UTC)
    author = git("config", "user.name") or getpass.getuser()
    entry = {
        "time_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": author,
        "name": config.name,
        "config_path": config.path,
        "config": config.as_dict(),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "src", "configs")),
        "csv_sha256": sha256,
        "seconds": round(seconds, 1),
        "diagnostics": diagnostics,
        "public_sanity": sanity,
        "note": note,
    }
    if external is not None:
        entry["external"] = external
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    safe_author = "".join(c if c.isalnum() else "-" for c in author.lower())
    path = RUNS_DIR / f"{now.strftime('%Y%m%dT%H%M%S')}_{safe_author}_{config.name}.json"
    path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    return path


def load_all() -> list[dict]:
    entries = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            entries.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return sorted(entries, key=lambda e: e.get("time_utc", ""))


def table(entries: list[dict]) -> str:
    header = (
        f"{'time (UTC)':<20} {'author':<12} {'name':<28} {'poss':>5} {'dur s':>6} "
        f"{'pub spF1*':>9} {'ext spF1':>8} {'commit':<8} note"
    )
    lines = [header, "-" * len(header)]
    for e in entries:
        d = e.get("diagnostics") or {}
        s = e.get("public_sanity") or {}
        x = e.get("external") or {}
        commit = (e.get("git_commit") or "") + ("+" if e.get("git_dirty") else "")
        lines.append(
            f"{e.get('time_utc', ''):<20} {e.get('author', '')[:12]:<12} "
            f"{e.get('name', '')[:28]:<28} {d.get('n_possessions', ''):>5} "
            f"{d.get('mean_possession_s', ''):>6} {s.get('spellF1', ''):>9} "
            f"{x.get('spellF1', ''):>8} {commit:<8} "
            f"{e.get('note', '')}"
        )
    lines.append("* public 5-minute window: sanity check only, not a ranking (EVALUATION.md)")
    lines.append("  ext spF1: another, labelled match (possession validate); ranks versions")
    return "\n".join(lines)
