"""Integration test: DSPy module.forward() must call writer.set_candidate().

These tests exercise the hook added to evolution.prompts.prompt_module.PromptSectionModule
and evolution.tools.tool_module.ToolDescModule, which is the load-bearing piece
that lets the trace writer know which candidate text produced each prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evolution.meta_harness.trace_writer import (
    TraceWriter,
    make_tracing_metric,
    set_active_writer,
)


@pytest.fixture(autouse=True)
def _tracing_on(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLUTION_TRACING", "1")
    yield
    set_active_writer(None)


class _FakePredictor:
    """Callable stand-in for dspy.ChainOfThought.

    Preserves the ``.predict.signature.instructions`` structure so the
    ``PromptSectionModule.section_text`` property (which reads from
    that path) returns the expected text.

    Supports dynamic mutation of instructions via
    ``predict.signature.instructions = new_text`` so tests can simulate
    GEPA candidate mutation.
    """

    def __init__(self, instructions: str = ""):
        self.predict = SimpleNamespace(
            signature=SimpleNamespace(instructions=instructions)
        )

    def __call__(self, **_kwargs):
        return SimpleNamespace(output="stubbed response")


def _fake_predictor(**_kwargs):
    """Legacy stand-in (function form) — used where structure isn't needed."""
    return SimpleNamespace(output="stubbed response")


def test_prompt_section_module_forward_calls_set_candidate(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module = PromptSectionModule("CANDIDATE TEXT V1")

    # Replace predictor with a structure-preserving stub so the
    # section_text property (which reads from .predict.signature.instructions)
    # still returns the right text.
    module.predictor = _FakePredictor(instructions="CANDIDATE TEXT V1")

    # Invoke forward — this should trigger writer.set_candidate
    pred = module.forward(task_input="What is the capital of France?")
    assert pred.output == "stubbed response"

    # Writer should now know about this candidate
    assert writer.iteration == 0

    # Now log an eval and confirm the candidate snapshot was written
    writer.log_eval(
        SimpleNamespace(task_input="capital?", expected_behavior="Paris"),
        pred,
        0.9,
    )
    candidate_file = tmp_path / "traces" / "iteration_000" / "candidate.txt"
    assert candidate_file.exists()
    # After Phase II refactor, candidate is just the signature instructions
    # (no composite). GEPA mutates signature.instructions directly.
    assert candidate_file.read_text() == "CANDIDATE TEXT V1"


def test_prompt_module_candidate_change_advances_iteration(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module_a = PromptSectionModule("VERSION A")
    module_a.predictor = _FakePredictor(instructions="VERSION A")
    module_a.forward(task_input="task 1")
    writer.log_eval(SimpleNamespace(task_input="t1", expected_behavior="e"), SimpleNamespace(output="o1"), 0.5)

    module_b = PromptSectionModule("VERSION B")  # GEPA would deep-copy like this
    module_b.predictor = _FakePredictor(instructions="VERSION B")
    module_b.forward(task_input="task 2")
    writer.log_eval(SimpleNamespace(task_input="t2", expected_behavior="e"), SimpleNamespace(output="o2"), 0.8)

    iter0_content = (tmp_path / "traces" / "iteration_000" / "candidate.txt").read_text()
    iter1_content = (tmp_path / "traces" / "iteration_001" / "candidate.txt").read_text()
    assert iter0_content == "VERSION A"
    assert iter1_content == "VERSION B"


def test_tool_desc_module_forward_calls_set_candidate(tmp_path: Path):
    from evolution.tools.tool_module import ToolDescModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module = ToolDescModule("search the web for a query string")
    # Structure-preserving stub — description_text is now a property
    # reading from predict.signature.instructions
    module.predictor = _FakePredictor(instructions="search the web for a query string")

    pred = module.forward(task_input="find recent news", available_tools="web_search, read_file")
    assert pred.output == "stubbed response"

    writer.log_eval(
        SimpleNamespace(task_input="find news", expected_behavior="web_search"),
        pred,
        1.0,
    )
    candidate_file = tmp_path / "traces" / "iteration_000" / "candidate.txt"
    assert candidate_file.exists()
    # After Phase II refactor: candidate.txt is just the description text,
    # no composite.
    assert candidate_file.read_text() == "search the web for a query string"


def test_forward_hook_is_no_op_without_writer(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    # No set_active_writer call — hook must be a silent no-op
    set_active_writer(None)

    module = PromptSectionModule("CANDIDATE")
    module.predictor = _fake_predictor

    # Should not raise
    pred = module.forward(task_input="hi")
    assert pred.output == "stubbed response"


def test_wrapped_metric_plus_forward_writes_full_trace(tmp_path: Path):
    """End-to-end: forward() → wrapped metric → trace file contains everything."""
    from evolution.prompts.prompt_module import PromptSectionModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    def inner_metric(ex, pred, trace=None):
        return 0.73

    wrapped = make_tracing_metric(inner_metric)

    module = PromptSectionModule("Act as a terse analyst.")
    module.predictor = _FakePredictor(instructions="Act as a terse analyst.")

    example = SimpleNamespace(task_input="summarize", expected_behavior="short summary", category="analysis")
    prediction = module.forward(task_input="summarize")
    score = wrapped(example, prediction)

    assert score == 0.73

    task_files = list((tmp_path / "traces" / "iteration_000" / "tasks").glob("*.json"))
    assert len(task_files) == 1
    trace = json.loads(task_files[0].read_text())
    assert trace["task_input"] == "summarize"
    assert trace["expected_behavior"] == "short summary"
    assert trace["agent_output"] == "stubbed response"
    assert trace["score"] == 0.73
    assert trace["category"] == "analysis"
    # After Phase II refactor, candidate_preview is the raw instructions,
    # not a composite — so it matches section_text exactly.
    assert trace["candidate_chars"] == len("Act as a terse analyst.")
    assert trace["candidate_preview"] == "Act as a terse analyst."
