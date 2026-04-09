"""Tests for evolution.meta_harness.trace_writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.meta_harness.trace_writer import (
    TraceWriter,
    get_active_writer,
    make_tracing_metric,
    set_active_writer,
    tracing_enabled,
)


@pytest.fixture(autouse=True)
def _tracing_on(monkeypatch):
    """Most tests exercise the enabled path; disable it where needed."""
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "1")
    yield
    set_active_writer(None)


def _make_example(task_input="what is 2+2?", expected_behavior="answer is 4", **extras):
    return SimpleNamespace(task_input=task_input, expected_behavior=expected_behavior, **extras)


def _make_prediction(output="the answer is 4"):
    return SimpleNamespace(output=output)


# ── Basic plumbing ─────────────────────────────────────────────────────


def test_tracing_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "0")
    assert tracing_enabled() is False
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "1")
    assert tracing_enabled() is True
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "true")
    assert tracing_enabled() is True
    monkeypatch.delenv("HERMES_EVOLUTION_TRACING", raising=False)
    assert tracing_enabled() is False


def test_writer_creates_archive_and_manifest(tmp_path: Path):
    writer = TraceWriter(tmp_path / "traces", artifact_name="TEST_SECTION")
    manifest_path = tmp_path / "traces" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_name"] == "TEST_SECTION"
    assert manifest["schema_version"] == 1


def test_active_writer_set_and_clear(tmp_path: Path):
    assert get_active_writer() is None
    writer = TraceWriter(tmp_path / "t")
    set_active_writer(writer)
    assert get_active_writer() is writer
    set_active_writer(None)
    assert get_active_writer() is None


def test_active_writer_visible_across_threads(tmp_path: Path):
    """Critical regression test: DSPy optimizers run in worker threads,
    so the active writer must be a process-global, not thread-local."""
    import threading

    writer = TraceWriter(tmp_path / "cross_thread")
    set_active_writer(writer)

    seen: list = []
    def worker():
        seen.append(get_active_writer())

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert len(seen) == 1
    assert seen[0] is writer, "writer must be visible from worker threads"
    set_active_writer(None)


# ── Per-task logging ───────────────────────────────────────────────────


def test_log_eval_writes_task_json(tmp_path: Path):
    writer = TraceWriter(tmp_path / "traces")
    writer.set_candidate("You are a helpful agent.")
    ex = _make_example(category="math", difficulty="easy")
    pred = _make_prediction("4")

    path = writer.log_eval(ex, pred, 0.95, feedback="Correct")

    assert path is not None and path.exists()
    trace = json.loads(path.read_text())
    assert trace["task_input"] == "what is 2+2?"
    assert trace["expected_behavior"] == "answer is 4"
    assert trace["agent_output"] == "4"
    assert trace["score"] == 0.95
    assert trace["feedback"] == "Correct"
    assert trace["category"] == "math"
    assert trace["difficulty"] == "easy"
    assert trace["iteration"] == 0
    assert trace["task_id"] == "task_00001"
    assert trace["candidate_chars"] == len("You are a helpful agent.")


def test_candidate_change_advances_iteration(tmp_path: Path):
    writer = TraceWriter(tmp_path / "traces")

    # Iteration 0
    writer.set_candidate("candidate v1")
    writer.log_eval(_make_example(), _make_prediction("a"), 0.5)
    writer.log_eval(_make_example(), _make_prediction("b"), 0.6)

    # Same candidate → stays in iteration 0
    writer.set_candidate("candidate v1")
    writer.log_eval(_make_example(), _make_prediction("c"), 0.7)

    # New candidate → iteration 1
    writer.set_candidate("candidate v2")
    writer.log_eval(_make_example(), _make_prediction("d"), 0.8)

    iter0 = tmp_path / "traces" / "iteration_000" / "tasks"
    iter1 = tmp_path / "traces" / "iteration_001" / "tasks"
    assert sorted(p.name for p in iter0.glob("*.json")) == [
        "task_00001.json", "task_00002.json", "task_00003.json",
    ]
    assert sorted(p.name for p in iter1.glob("*.json")) == ["task_00001.json"]

    # Candidate snapshot per iteration
    assert (tmp_path / "traces" / "iteration_000" / "candidate.txt").read_text() == "candidate v1"
    assert (tmp_path / "traces" / "iteration_001" / "candidate.txt").read_text() == "candidate v2"


def test_task_counter_resets_per_iteration(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t")
    writer.set_candidate("a")
    writer.log_eval(_make_example(), _make_prediction(), 0.1)
    writer.log_eval(_make_example(), _make_prediction(), 0.2)
    writer.set_candidate("b")  # new iteration
    p = writer.log_eval(_make_example(), _make_prediction(), 0.3)
    assert p is not None
    trace = json.loads(p.read_text())
    assert trace["task_id"] == "task_00001"
    assert trace["iteration"] == 1


def test_large_agent_output_is_truncated(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t", max_task_bytes=1000)
    huge_output = "x" * 10_000
    writer.set_candidate("c")
    path = writer.log_eval(_make_example(), _make_prediction(huge_output), 0.5)
    trace = json.loads(path.read_text())
    assert "truncated" in trace["agent_output"]
    assert len(trace["agent_output"].encode("utf-8")) < 2000  # well under the full 10K


def test_log_eval_is_safe_when_example_missing_fields(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t")
    writer.set_candidate("c")
    empty = SimpleNamespace()  # no task_input, no expected_behavior
    empty_pred = SimpleNamespace()  # no output
    path = writer.log_eval(empty, empty_pred, 0.0)
    assert path is not None
    trace = json.loads(path.read_text())
    assert trace["task_input"] == ""
    assert trace["agent_output"] == ""
    assert trace["score"] == 0.0


def test_log_eval_never_raises_on_bad_input(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t")
    # No set_candidate call — should still work
    # Deliberately broken example that will raise if accessed carelessly

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    # Should not raise — TraceWriter wraps everything in try/except
    path = writer.log_eval(Broken(), Broken(), 0.5)
    # It may or may not write a file; the important thing is no exception


# ── Metric wrapping ────────────────────────────────────────────────────


def test_make_tracing_metric_passthrough_when_no_writer(tmp_path: Path, monkeypatch):
    calls = []

    def inner(ex, pred, trace=None):
        calls.append((ex, pred))
        return 0.77

    wrapped = make_tracing_metric(inner)
    set_active_writer(None)
    score = wrapped(_make_example(), _make_prediction(), trace=None)
    assert score == 0.77
    assert len(calls) == 1
    assert wrapped.__wrapped__ is inner


def test_make_tracing_metric_logs_when_writer_active(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t")
    writer.set_candidate("cand")
    set_active_writer(writer)

    def inner(ex, pred, trace=None):
        return 0.42

    wrapped = make_tracing_metric(inner)
    score = wrapped(_make_example(task_input="hi"), _make_prediction("hello"), trace=None)

    assert score == 0.42
    files = list((tmp_path / "t" / "iteration_000" / "tasks").glob("*.json"))
    assert len(files) == 1
    trace = json.loads(files[0].read_text())
    assert trace["task_input"] == "hi"
    assert trace["agent_output"] == "hello"
    assert trace["score"] == 0.42


def test_make_tracing_metric_skips_when_env_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "0")
    writer = TraceWriter(tmp_path / "t")
    writer.set_candidate("cand")
    set_active_writer(writer)

    def inner(ex, pred, trace=None):
        return 0.5

    wrapped = make_tracing_metric(inner)
    score = wrapped(_make_example(), _make_prediction())
    assert score == 0.5
    # No task files should have been written
    assert not list((tmp_path / "t").glob("iteration_*/tasks/*.json"))


def test_make_tracing_metric_preserves_extra_args(tmp_path: Path):
    writer = TraceWriter(tmp_path / "t")
    writer.set_candidate("cand")
    set_active_writer(writer)

    captured = {}

    def inner(ex, pred, trace=None, weights=None):
        captured["weights"] = weights
        return 0.9

    wrapped = make_tracing_metric(inner)
    wrapped(_make_example(), _make_prediction(), None, weights={"a": 1})
    assert captured["weights"] == {"a": 1}
