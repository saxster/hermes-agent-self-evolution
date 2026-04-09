"""Tests for evolution.meta_harness.scoped_tools.

These focus on the context-manager behavior (env-var save/restore,
nesting, error handling). The underlying enforcement is tested in
hermes-agent/tests/tools/test_file_read_safety.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evolution.meta_harness.scoped_tools import (
    current_scope,
    scoped_reads,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_READ_SAFE_ROOT", raising=False)
    yield


def test_scoped_reads_sets_env(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()

    assert os.environ.get("HERMES_READ_SAFE_ROOT") is None
    with scoped_reads(archive) as resolved:
        assert os.environ["HERMES_READ_SAFE_ROOT"] == str(archive.resolve())
        assert resolved == archive.resolve()
    assert os.environ.get("HERMES_READ_SAFE_ROOT") is None


def test_scoped_reads_restores_previous_value(tmp_path: Path):
    original = tmp_path / "original"
    new_scope = tmp_path / "new"
    original.mkdir()
    new_scope.mkdir()

    os.environ["HERMES_READ_SAFE_ROOT"] = str(original)
    try:
        with scoped_reads(new_scope):
            assert os.environ["HERMES_READ_SAFE_ROOT"] == str(new_scope.resolve())
        # Restored to the pre-block value
        assert os.environ["HERMES_READ_SAFE_ROOT"] == str(original)
    finally:
        os.environ.pop("HERMES_READ_SAFE_ROOT", None)


def test_scoped_reads_restores_on_exception(tmp_path: Path):
    archive = tmp_path / "a"
    archive.mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        with scoped_reads(archive):
            assert "HERMES_READ_SAFE_ROOT" in os.environ
            raise RuntimeError("boom")
    assert "HERMES_READ_SAFE_ROOT" not in os.environ


def test_scoped_reads_rejects_nonexistent_dir(tmp_path: Path):
    nope = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        with scoped_reads(nope):
            pass  # pragma: no cover


def test_scoped_reads_rejects_file_not_dir(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        with scoped_reads(f):
            pass  # pragma: no cover


def test_scoped_reads_nested(tmp_path: Path):
    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()

    with scoped_reads(outer):
        assert os.environ["HERMES_READ_SAFE_ROOT"] == str(outer.resolve())
        with scoped_reads(inner):
            assert os.environ["HERMES_READ_SAFE_ROOT"] == str(inner.resolve())
        # Outer restored after inner exits
        assert os.environ["HERMES_READ_SAFE_ROOT"] == str(outer.resolve())
    assert "HERMES_READ_SAFE_ROOT" not in os.environ


def test_current_scope_reflects_env(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    assert current_scope() is None
    with scoped_reads(archive):
        scope = current_scope()
        assert scope is not None
        assert scope == archive.resolve()
    assert current_scope() is None


def test_scoped_reads_expands_user_tilde(tmp_path: Path, monkeypatch):
    """Pass a string with ~; context manager should expand it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "archive").mkdir()
    with scoped_reads("~/archive") as resolved:
        assert resolved == (tmp_path / "archive").resolve()


def test_scoped_reads_tilde_in_env_respected(tmp_path: Path):
    """Verify the resolved env value is usable for _is_read_denied downstream."""
    archive = tmp_path / "archive"
    target = archive / "inside.txt"
    outside = tmp_path / "outside.txt"
    archive.mkdir()
    target.write_text("inside")
    outside.write_text("outside")

    # Import the hermes-agent function and verify it honors our env var
    import sys

    hermes_agent_root = Path(__file__).parent.parent.parent.parent.parent / "hermes-agent"
    if not hermes_agent_root.exists():
        pytest.skip(f"hermes-agent not found at {hermes_agent_root}")
    sys.path.insert(0, str(hermes_agent_root))
    try:
        from tools.file_operations import _is_read_denied  # type: ignore
    finally:
        sys.path.pop(0)

    with scoped_reads(archive):
        denied_inside, _ = _is_read_denied(str(target))
        denied_outside, _ = _is_read_denied(str(outside))
        assert denied_inside is False
        assert denied_outside is True
