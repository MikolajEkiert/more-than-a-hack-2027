"""The agent hooks: protection is always on; the data guard blocks only when switched on."""

import json
import subprocess
import sys

import pytest

from possession._paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import agent_hooks  # noqa: E402

HOOK = REPO_ROOT / "scripts" / "agent_hooks.py"


def call(event, tool, payload):
    return subprocess.run(
        [sys.executable, str(HOOK), event, "--tool", tool],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )


def test_protected_edit_blocked_for_claude():
    path = str(REPO_ROOT / "evaluation" / "possession_metrics.py")
    result = call("pre", "claude", {"tool_name": "Edit", "tool_input": {"file_path": path}})
    assert result.returncode == 2 and "organiser-owned" in result.stderr


def test_own_code_edit_allowed():
    path = str(REPO_ROOT / "src" / "possession" / "cli.py")
    assert (
        call("pre", "claude", {"tool_name": "Edit", "tool_input": {"file_path": path}}).returncode
        == 0
    )


def test_session_context_formats():
    cursor = json.loads(call("session", "cursor", {}).stdout)
    assert "AGENTS.md" in cursor["additional_context"]
    assert "AGENTS.md" in call("session", "claude", {}).stdout


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(agent_hooks, "guard_on", lambda: True)


def test_guard_off_allows_everything(monkeypatch):
    monkeypatch.setattr(agent_hooks, "guard_on", lambda: False)
    assert agent_hooks.read_reason(str(REPO_ROOT / "data" / "tactical" / "x_atd.txt")) is None
    assert agent_hooks.shell_reason("head data/tactical/x_atd.txt") is None


@pytest.mark.usefixtures("guard")
class TestGuardOn:
    def test_blocks_data_and_derived_files(self):
        for path in (
            "data/tactical/BAR-ATH_tactical_atd.txt",
            ".cache/BAR-ATH_match.npz",
            "submission/team.csv",
            "evaluation/results/mine.csv",
        ):
            assert agent_hooks.read_reason(str(REPO_ROOT / path)), path

    def test_allows_docs_and_baselines(self):
        for path in ("data/README.md", "evaluation/results/baseline_smooth50.csv", "AGENTS.md"):
            assert agent_hooks.read_reason(str(REPO_ROOT / path)) is None, path

    def test_blocks_printing_data_in_shell(self):
        for cmd in (
            "head -n 5 data/tactical/BAR-ATH_tactical_atd.txt",
            "uv run possession run configs/x.toml | cat data/x",
            "python -c 'from possession.loader import load_match; print(load_match().ball)'",
        ):
            assert agent_hooks.shell_reason(cmd), cmd

    def test_allows_aggregate_tooling(self):
        for cmd in (
            "uv run possession run configs/baseline_smooth50.toml",
            "uv run pytest -q",
            "uv run possession inspect ball",
            "uv run possession inspect labels --csv evaluation/results/mine.csv --json",
            "make check",
            "git status",
        ):
            assert agent_hooks.shell_reason(cmd) is None, cmd
