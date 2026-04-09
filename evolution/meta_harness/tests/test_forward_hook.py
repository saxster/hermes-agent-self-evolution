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


def _fake_predictor(**_kwargs):
    """Stand-in for dspy.ChainOfThought that returns a canned output."""
    return SimpleNamespace(output="stubbed response")


def test_prompt_section_module_forward_calls_set_candidate(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module = PromptSectionModule("CANDIDATE TEXT V1")

    # Replace the expensive ChainOfThought predictor with a stub
    module.predictor = _fake_predictor

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
    # Candidate is a composite "section_text:\n...\n\npredictor_instructions:\n..."
    # so we check containment rather than equality.
    content = candidate_file.read_text()
    assert "CANDIDATE TEXT V1" in content
    assert "section_text:" in content
    assert "predictor_instructions:" in content


def test_prompt_module_candidate_change_advances_iteration(tmp_path: Path):
    from evolution.prompts.prompt_module import PromptSectionModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module_a = PromptSectionModule("VERSION A")
    module_a.predictor = _fake_predictor
    module_a.forward(task_input="task 1")
    writer.log_eval(SimpleNamespace(task_input="t1", expected_behavior="e"), SimpleNamespace(output="o1"), 0.5)

    module_b = PromptSectionModule("VERSION B")  # GEPA would deep-copy like this
    module_b.predictor = _fake_predictor
    module_b.forward(task_input="task 2")
    writer.log_eval(SimpleNamespace(task_input="t2", expected_behavior="e"), SimpleNamespace(output="o2"), 0.8)

    iter0_content = (tmp_path / "traces" / "iteration_000" / "candidate.txt").read_text()
    iter1_content = (tmp_path / "traces" / "iteration_001" / "candidate.txt").read_text()
    assert "VERSION A" in iter0_content
    assert "VERSION B" in iter1_content


def test_tool_desc_module_forward_calls_set_candidate(tmp_path: Path):
    from evolution.tools.tool_module import ToolDescModule

    writer = TraceWriter(tmp_path / "traces")
    set_active_writer(writer)

    module = ToolDescModule("search the web for a query string")
    module.predictor = _fake_predictor

    pred = module.forward(task_input="find recent news", available_tools="web_search, read_file")
    assert pred.output == "stubbed response"

    writer.log_eval(
        SimpleNamespace(task_input="find news", expected_behavior="web_search"),
        pred,
        1.0,
    )
    candidate_file = tmp_path / "traces" / "iteration_000" / "candidate.txt"
    assert candidate_file.exists()
    content = candidate_file.read_text()
    assert "search the web for a query string" in content
    assert "description_text:" in content
    assert "predictor_instructions:" in content


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
    module.predictor = _fake_predictor

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
    # candidate_chars now reflects the composite (section_text + instructions),
    # which is longer than section_text alone — just assert it's >= that size.
    assert trace["candidate_chars"] >= len("Act as a terse analyst.")
    assert trace["candidate_preview"].startswith("section_text:")
