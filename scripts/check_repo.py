"""Repository guard, run by pre-commit and CI. Standard library only.

1. Organiser-owned files are unchanged. The scorer, validator, public GT, committed baselines and
   starter modules are the specification; an agent "fixing" them would silently change what our
   numbers mean. Their hashes live in scripts/protected.sha256.
2. Nothing that must not be public is tracked: match data, video, parsed caches, submission files,
   notebook outputs (derived data, RULES.md "The data"), or large blobs.

    python scripts/check_repo.py                    # check (exit 1 on any problem)
    python scripts/check_repo.py --update-manifest  # after deliberately pulling an upstream update
"""

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "protected.sha256"
PROTECTED = [
    "evaluation/possession_metrics.py",
    "evaluation/run_evaluation.py",
    "evaluation/validate_submission.py",
    "evaluation/gt/*",
    "evaluation/results/baseline_*.csv",
    "starter/*.py",
]
FORBIDDEN = [
    ("data/*", "match data is not redistributable", ["data/README.md"]),
    ("*.mp4", "video", []),
    (".cache/*", "parsed match arrays", []),
    ("submission/*", "submission files are a full set of labels", []),
    ("evaluation/results/*.csv", "submissions stay out of git", ["evaluation/results/baseline_*"]),
    ("*_possession_gt.csv", "private ground truth", []),
]
MAX_BYTES = 5_000_000


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if (ROOT / line).exists()]


def protected_files() -> list[str]:
    files = set()
    for pattern in PROTECTED:
        files.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_protected() -> list[str]:
    if not MANIFEST.is_file():
        return [f"{MANIFEST.relative_to(ROOT)} missing; run with --update-manifest"]
    expected = dict(line.split(None, 1)[::-1] for line in MANIFEST.read_text().splitlines())
    problems = []
    for path, sha in sorted(expected.items()):
        if not (ROOT / path).is_file():
            problems.append(f"protected file deleted: {path}")
        elif digest(path) != sha:
            problems.append(f"protected file modified: {path} (organiser-owned, do not edit)")
    for path in protected_files():
        if path not in expected:
            problems.append(f"new file in a protected location: {path}")
    return problems


def check_hygiene(files: list[str]) -> list[str]:
    problems = []
    for path in files:
        for pattern, why, allowed in FORBIDDEN:
            if fnmatch.fnmatch(path, pattern) and not any(
                fnmatch.fnmatch(path, a) for a in allowed
            ):
                problems.append(f"must not be committed: {path} ({why})")
        if (ROOT / path).stat().st_size > MAX_BYTES and not path.endswith(".lock"):
            problems.append(f"file over {MAX_BYTES // 1_000_000} MB: {path}")
        if path.endswith(".ipynb") and notebook_has_outputs(ROOT / path):
            problems.append(f"notebook has outputs: {path} (run: uv run nbstripout {path})")
    return problems


def notebook_has_outputs(path: Path) -> bool:
    try:
        cells = json.loads(path.read_text()).get("cells", [])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return any(c.get("outputs") for c in cells if c.get("cell_type") == "code")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()

    if args.update_manifest:
        lines = [f"{digest(p)}  {p}" for p in protected_files()]
        MANIFEST.write_text("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} hashes to {MANIFEST.relative_to(ROOT)}")
        return 0

    problems = check_protected() + check_hygiene(tracked())
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print("OK    protected files intact, nothing private tracked")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
