"""The repo guard rejects a manifest that differs from main's unless the change is deliberate."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_repo  # noqa: E402


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "scripts" / "protected.sha256"
    path.parent.mkdir()
    path.write_text("aaa  evaluation/possession_metrics.py\n")
    monkeypatch.setattr(check_repo, "ROOT", tmp_path)
    monkeypatch.setattr(check_repo, "MANIFEST", path)
    monkeypatch.delenv("ALLOW_MANIFEST_CHANGE", raising=False)
    return path


def test_manifest_equal_to_main_passes(manifest, monkeypatch):
    monkeypatch.setattr(check_repo, "manifest_on", lambda ref: manifest.read_text())
    assert check_repo.check_manifest_unchanged() == []


def test_manifest_changed_against_main_fails(manifest, monkeypatch):
    monkeypatch.setattr(check_repo, "manifest_on", lambda ref: "bbb  evaluation/x.py\n")
    problems = check_repo.check_manifest_unchanged()
    assert len(problems) == 1 and "differs from origin/main" in problems[0]


def test_deliberate_manifest_change_is_allowed(manifest, monkeypatch):
    monkeypatch.setattr(check_repo, "manifest_on", lambda ref: "bbb  evaluation/x.py\n")
    monkeypatch.setenv("ALLOW_MANIFEST_CHANGE", "1")
    assert check_repo.check_manifest_unchanged() == []


def test_no_main_to_compare_passes(manifest, monkeypatch):
    monkeypatch.setattr(check_repo, "manifest_on", lambda ref: None)
    assert check_repo.check_manifest_unchanged() == []
