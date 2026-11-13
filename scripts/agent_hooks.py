"""Hooks shared by Claude Code (.claude/settings.json) and Cursor (.cursor/hooks.json).

One script, both tools: each event reads the tool's JSON on stdin and answers in that tool's
format, so the behaviour is identical whichever editor a teammate uses. Standard library only.

    agent_hooks.py session  --tool claude|cursor   inject branch, task, recent experiments
    agent_hooks.py pre      --tool claude          before Read/Edit/Write/Grep/Bash (Claude)
    agent_hooks.py read     --tool cursor          beforeReadFile (Cursor)
    agent_hooks.py shell    --tool cursor          beforeShellExecution (Cursor)
    agent_hooks.py format   --tool claude|cursor   after an edit: ruff format + fix

Two policies:
  * PROTECT (always on): organiser-owned files (see scripts/check_repo.py) are not edited.
  * DATA GUARD (scripts/agent_guard.json, off by default): when on, agents may not read the match
    data, its parsed cache, submission files, or notebooks with outputs, nor run shell commands
    that print them. Turn it on with `make guard-on` once the official dataset is in data/
    (FAQ.md: the terms forbid giving the dataset or extracts of it to a cloud-hosted LLM).
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_repo import PROTECTED, notebook_has_outputs  # noqa: E402

GUARD_FILE = ROOT / "scripts" / "agent_guard.json"
RUNS_DIR = ROOT / "experiments" / "runs"
FORMAT_EXCLUDE = ("evaluation/", "starter/", "games.py")

SENSITIVE_SHELL = re.compile(
    r"(\bdata/|\.cache\b|\bsubmission/|MTAH_DATA_DIR|\.mp4\b|_atd\.(txt|xml)|_homography\.csv"
    r"|_smart_tagging\.xml|evaluation/results/(?!baseline_))"
)
INLINE_PYTHON = re.compile(r"python[0-9.]*\s+(-c\b|-\s|<<)|ipython|jupyter\s+(console|nbconvert)")
DATA_API = re.compile(r"load_match|asset_paths|games\.require|np\.load|read_csv|kloppy|\.npz")
SAFE_PREFIXES = (
    "uv run possession",
    "uv run pytest",
    "make ",
    "git ",
    "ls ",
    "du ",
    "python scripts/check_repo.py",
    "uv run python scripts/check_repo.py",
)


# -- policy ------------------------------------------------------------------------------------
def guard_on() -> bool:
    try:
        return bool(json.loads(GUARD_FILE.read_text()).get("enabled"))
    except (OSError, json.JSONDecodeError):
        return False


def rel(path: str) -> str | None:
    """Repo-relative POSIX path, or None if outside the repository."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return None


def protected_reason(path: str) -> str | None:
    r = rel(path)
    if r and any(fnmatch.fnmatch(r, pattern) for pattern in PROTECTED):
        return (
            f"{r} is organiser-owned (the scoring spec) and must not be edited. Import from it; "
            "put new code in src/possession/. If upstream changed it, a human pulls the update and "
            "runs `python scripts/check_repo.py --update-manifest`."
        )
    return None


def read_reason(path: str) -> str | None:
    if not guard_on():
        return None
    resolved = Path(path).resolve()
    data_dirs = [ROOT / "data"]
    if os.environ.get("MTAH_DATA_DIR"):
        data_dirs.append(Path(os.environ["MTAH_DATA_DIR"]).resolve())
    r = rel(path) or ""
    if r == "data/README.md":
        return None
    if any(resolved.is_relative_to(d) for d in data_dirs) or r.split("/")[0] in (
        ".cache",
        "submission",
    ):
        return f"DATA GUARD: {r or resolved} is match data; agents never read it directly."
    if fnmatch.fnmatch(r, "evaluation/results/*.csv") and "/baseline_" not in r:
        return "DATA GUARD: submission CSVs are frame-level labels; use `uv run possession log`."
    if r.endswith(".ipynb") and resolved.is_file() and notebook_has_outputs(resolved):
        return f"DATA GUARD: {r} has outputs (data extracts). Ask a human to clear them first."
    return None


def shell_reason(command: str) -> str | None:
    if not guard_on():
        return None
    stripped = command.strip()
    if stripped.startswith(SAFE_PREFIXES) and not re.search(r"[|;&]|\bcat\b|\bhead\b", stripped):
        return None
    if SENSITIVE_SHELL.search(command):
        return (
            "DATA GUARD: this command touches match data. Write code in src/possession/ and run it "
            "through `uv run possession run ...` or `uv run pytest`, which print aggregates only."
        )
    if INLINE_PYTHON.search(command) and DATA_API.search(command):
        return (
            "DATA GUARD: inline Python that loads match data could print raw rows. Put the logic "
            "in a module or test and print aggregates only (counts, rates, scores)."
        )
    return None


# -- session context ---------------------------------------------------------------------------
def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def frontmatter(path: Path) -> dict:
    meta = {}
    lines = path.read_text().splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    return meta


def session_context() -> str:
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    out = ["# Session context (auto-injected by scripts/agent_hooks.py)", ""]
    out.append(
        f"- branch: `{branch}`; uncommitted files: {len(git('status', '--porcelain').splitlines())}"
    )
    out.append(
        f"- data guard: {'ON: never read data/, .cache/, submissions' if guard_on() else 'off'}"
    )

    match = re.match(r"task/(\d+)", branch)
    task_files = sorted((ROOT / "tasks").glob("[0-9]*.md"))
    current = [p for p in task_files if match and p.name.startswith(match.group(1) + "-")]
    if current:
        body = current[0].read_text().splitlines()
        out += ["", f"## Current task: tasks/{current[0].name}", "", *body[:80]]
    elif branch in ("main", "master"):
        out.append("- you are on main: start work on a task branch (`/do-task NN`), not here")

    active = []
    for p in task_files:
        meta = frontmatter(p)
        if meta.get("status") in ("doing", "review"):
            active.append(f"  - {p.name}: {meta.get('status')} ({meta.get('owner') or 'unowned'})")
    if active:
        out += ["", "## Tasks in progress (do not duplicate them)", *active]

    runs = sorted(RUNS_DIR.glob("*.json"))[-5:]
    if runs:
        out += ["", "## Latest experiments (`uv run possession log` for all)"]
        for p in runs:
            try:
                e = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            d, s = e.get("diagnostics") or {}, e.get("public_sanity") or {}
            out.append(
                f"  - {e.get('name')} by {e.get('author')}: {d.get('n_possessions')} poss, "
                f"{d.get('mean_possession_s')} s mean, public-sanity spellF1 {s.get('spellF1')}"
                f" -- {e.get('note') or ''}"
            )
    out += ["", "Read AGENTS.md before acting. Definition of done: `make check` passes."]
    return "\n".join(out)


# -- formatting --------------------------------------------------------------------------------
def format_file(path: str) -> str:
    """ruff format + fix one file; returns remaining lint errors (empty if clean)."""
    r = rel(path)
    uv = shutil.which("uv")
    if not r or not r.endswith(".py") or r.startswith(FORMAT_EXCLUDE) or not uv:
        return ""
    run = {"cwd": ROOT, "capture_output": True, "text": True, "timeout": 60}
    subprocess.run([uv, "run", "--quiet", "ruff", "format", r], **run)
    result = subprocess.run([uv, "run", "--quiet", "ruff", "check", "--fix", "--quiet", r], **run)
    return result.stdout.strip() if result.returncode else ""


# -- entry point -------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["session", "pre", "read", "shell", "format"])
    parser.add_argument("--tool", choices=["claude", "cursor"], required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}

    if args.event == "session":
        context = session_context()
        print(json.dumps({"additional_context": context}) if args.tool == "cursor" else context)
        return 0

    if args.event == "pre":  # Claude PreToolUse: exit 2 blocks the tool and shows stderr to Claude
        tool, data = payload.get("tool_name", ""), payload.get("tool_input") or {}
        path = data.get("file_path") or data.get("notebook_path") or data.get("path") or ""
        reason = None
        if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit") and path:
            reason = protected_reason(path)
        if not reason and tool in ("Read", "Grep", "NotebookEdit") and path:
            reason = read_reason(path)
        if not reason and tool == "Bash":
            reason = shell_reason(data.get("command", ""))
        if reason:
            print(reason, file=sys.stderr)
            return 2
        return 0

    if args.event == "read":  # Cursor beforeReadFile
        reason = read_reason(payload.get("file_path", ""))
        answer = (
            {"permission": "deny", "user_message": reason} if reason else {"permission": "allow"}
        )
        print(json.dumps(answer))
        return 0

    if args.event == "shell":  # Cursor beforeShellExecution
        reason = shell_reason(payload.get("command", ""))
        answer = (
            {"permission": "deny", "user_message": reason, "agent_message": reason}
            if reason
            else {"permission": "allow"}
        )
        print(json.dumps(answer))
        return 0

    if args.event == "format":
        path = (payload.get("tool_input") or {}).get("file_path") or payload.get("file_path", "")
        errors = format_file(path) if path else ""
        if errors and args.tool == "claude":  # PostToolUse exit 2: stderr goes back to Claude
            print(
                f"ruff found problems it could not fix in {rel(path)}:\n{errors}", file=sys.stderr
            )
            return 2
        if args.tool == "cursor":
            print("{}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
