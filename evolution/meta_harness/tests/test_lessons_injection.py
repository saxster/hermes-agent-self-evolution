"""Tests for Phase D: lessons injection into the GEPA proposer.

Covers:
- set_active_lessons / get_active_lessons / load_lessons_from_path
- make_tracing_metric's on_iteration_complete callback
- PromptSectionModule.forward() reading active lessons
- ToolDescModule.forward() reading active lessons
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evolution.meta_harness.trace_writer import (
    TraceWriter,
    get_active_lessons,
    load_lessons_from_path,
    make_tracing_metric,
    set_active_lessons,
    set_active_writer,
)


@pytest.fixture(autouse=True)
def _tracing_on(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "1")
    yield
    set_active_writer(None)
    set_active_lessons("")


# ── Lessons plumbing ───────────────────────────────────────────────────


def test_get_active_lessons_default_empty():
    set_active_lessons("")
    assert get_active_lessons() == ""


def test_set_and_get_active_lessons():
    set_active_lessons("## Failure patterns\n- test")
    assert "Failure patterns" in get_active_lessons()
    set_active_lessons("")
    assert get_active_lessons() == ""


def test_load_lessons_from_path_reads_file(tmp_path: Path):
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Learned\n- iteration 002 was best")

    loaded = load_lessons_from_path(lessons_file)
    assert "iteration 002" in loaded
    assert get_active_lessons() == loaded


def test_load_lessons_from_path_missing_file_is_empty(tmp_path: Path):
    missing = tmp_path / "nope.md"
    loaded = load_lessons_from_path(missing)
    assert loaded == ""
    assert get_active_lessons() == ""


def test_set_active_lessons_none_safe():
    set_active_lessons("previous")
    set_active_lessons(None)  # type: ignore[arg-type]
    assert get_active_lessons() == ""


# ── on_iteration_complete callback ─────────────────────────────────────


def test_callback_fires_when_iteration_advances(tmp_path: Path):
    """The callback should fire once per iteration boundary, with the
    index of the FINISHED iteration."""
    writer = TraceWriter(tmp_path / "t")
    set_active_writer(writer)

    calls: list[int] = []

    def inner_metric(ex, pred, trace=None):
        return 0.5

    def cb(finished_iter: int):
        calls.append(finished_iter)

    wrapped = make_tracing_metric(inner_metric, on_iteration_complete=cb)

    # Iteration 0
    writer.set_candidate("v1")
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    assert calls == []  # still iteration 0

    # Iteration 1 (new candidate)
    writer.set_candidate("v2")
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    # The first call in iteration 1 should fire cb(0)
    assert calls == [0]

    # More evals within iteration 1 — no duplicate fires
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    assert calls == [0]

    # Iteration 2
    writer.set_candidate("v3")
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    assert calls == [0, 1]


def test_callback_exception_does_not_crash_optimization(tmp_path: Path):
    """If the diagnosis callback raises, the metric still returns the score."""
    writer = TraceWriter(tmp_path / "t")
    set_active_writer(writer)

    def inner(ex, pred, trace=None):
        return 0.73

    def crash(_i):
        raise RuntimeError("diagnosis exploded")

    wrapped = make_tracing_metric(inner, on_iteration_complete=crash)

    writer.set_candidate("a")
    wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    writer.set_candidate("b")
    # This second call triggers the (crashing) callback — must still return 0.73
    score = wrapped(SimpleNamespace(task_input="t", expected_behavior="e"), SimpleNamespace(output="o"))
    assert score == 0.73


def test_callback_not_invoked_when_no_writer():
    """With no active writer, the callback must not fire at all."""
    set_active_writer(None)
    calls: list[int] = []

    def inner(ex, pred, trace=None):
        return 0.1

    wrapped = make_tracing_metric(inner, on_iteration_complete=lambda i: calls.append(i))
    wrapped(SimpleNamespace(task_input="x", expected_behavior="y"), SimpleNamespace(output="z"))
    assert calls == []


# ── DSPy module reads active lessons in forward() ─────────────────────


def test_prompt_section_module_reads_lessons(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    set_active_lessons("## Failure patterns\n- tasks 003,007 need Risk Dashboard")

    captured: dict = {}

    class _FakePred:
        def __init__(self):
            self.predict = SimpleNamespace(
                signature=SimpleNamespace(instructions="Base prompt")
            )
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output="stub")

    module = PromptSectionModule("Base prompt")
    module.predictor = _FakePred()
    module.forward(task_input="hello")

    assert "prior_lessons" in captured
    assert "Risk Dashboard" in captured["prior_lessons"]
    # After Phase II refactor, system_guidance is NOT a kwarg anymore —
    # section_text is the predictor's signature.instructions. The forward
    # call only passes prior_lessons + task_input.
    assert "system_guidance" not in captured
    assert captured["task_input"] == "hello"
    # Verify section_text property reads through to the instructions
    assert module.section_text == "Base prompt"


def test_prompt_section_module_empty_lessons_when_none_set(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    set_active_lessons("")

    captured: dict = {}

    class _FakePred:
        def __init__(self):
            self.predict = SimpleNamespace(
                signature=SimpleNamespace(instructions="Base")
            )
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output="stub")

    module = PromptSectionModule("Base")
    module.predictor = _FakePred()
    module.forward(task_input="hi")

    assert captured["prior_lessons"] == ""


def test_tool_desc_module_reads_lessons(tmp_path: Path):
    from evolution.tools.tool_module import ToolDescModule

    set_active_lessons("## Candidate guidance\n- mention 'regex' in description")

    captured: dict = {}

    class _FakePred:
        def __init__(self):
            self.predict = SimpleNamespace(
                signature=SimpleNamespace(instructions="search files")
            )
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output="stub")

    module = ToolDescModule("search files")
    module.predictor = _FakePred()
    module.forward(task_input="find something", available_tools="a, b, c")

    assert "prior_lessons" in captured
    assert "regex" in captured["prior_lessons"]
    # After Phase II refactor: tool_description is NOT a kwarg anymore —
    # the description IS the predictor's signature.instructions.
    assert "tool_description" not in captured
    assert captured["available_tools"] == "a, b, c"
    assert module.description_text == "search files"


def test_forward_hook_isolates_lessons_per_run():
    """Setting lessons then clearing should not leak."""
    from evolution.prompts.prompt_module import PromptSectionModule

    set_active_lessons("old stuff")
    set_active_lessons("")

    captured: dict = {}

    def fake_predictor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output="stub")

    module = PromptSectionModule("x")
    module.predictor = fake_predictor
    module.forward(task_input="t")
    assert captured["prior_lessons"] == ""
