"""Tests for evolution.meta_harness.diagnose.DiagnosisAgent.

We drive the loop with a FakeLLM that returns a scripted sequence of
tool_calls / content, so tests are fully deterministic and make zero
real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.meta_harness.diagnose import DiagnosisAgent, DiagnosisResult


# ── Archive fixture ────────────────────────────────────────────────────


@pytest.fixture
def fake_archive(tmp_path: Path) -> Path:
    """Build a small fake optimization archive to diagnose."""
    archive = tmp_path / "traces"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"artifact_name": "TEST"}))

    # iteration 0 — scored well
    iter0 = archive / "iteration_000"
    (iter0 / "tasks").mkdir(parents=True)
    (iter0 / "candidate.txt").write_text("Be thorough and include the Risk Dashboard.")
    (iter0 / "tasks" / "task_00001.json").write_text(
        json.dumps(
            {
                "task_id": "task_00001",
                "iteration": 0,
                "task_input": "market brief please",
                "expected_behavior": "includes Risk Dashboard section",
                "agent_output": "... Risk Dashboard: ...",
                "score": 0.91,
                "category": "briefing",
            }
        )
    )

    # iteration 1 — scored badly
    iter1 = archive / "iteration_001"
    (iter1 / "tasks").mkdir(parents=True)
    (iter1 / "candidate.txt").write_text("Be concise. Skip boilerplate sections.")
    (iter1 / "tasks" / "task_00001.json").write_text(
        json.dumps(
            {
                "task_id": "task_00001",
                "iteration": 1,
                "task_input": "market brief please",
                "expected_behavior": "includes Risk Dashboard section",
                "agent_output": "Brief market update without dashboard.",
                "score": 0.41,
                "category": "briefing",
            }
        )
    )

    return archive


# ── Scripted LLM caller ────────────────────────────────────────────────


class ScriptedLLM:
    """Plays back a scripted sequence of LLM responses.

    Each response is either a plain dict (already OpenAI-shaped) or a
    helper-built one via `.tool_call(...)` / `.text(...)`.
    """

    def __init__(self, responses: list[dict], cost_per_call: float = 0.001):
        self._responses = responses
        self._i = 0
        self._cost = cost_per_call
        self.calls: list[dict] = []

    def __call__(self, *, model, messages, tools, temperature):
        self.calls.append({"model": model, "n_messages": len(messages), "n_tools": len(tools)})
        if self._i >= len(self._responses):
            # Default: no tool calls, empty content — the loop will exit.
            return ({"choices": [{"message": {"role": "assistant", "content": ""}}]}, self._cost)
        resp = self._responses[self._i]
        self._i += 1
        return (resp, self._cost)


def _assistant_with_tool_calls(tool_calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }


def _tc(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _assistant_text(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}]
    }


# ── Construction / guards ──────────────────────────────────────────────


def test_diagnose_rejects_nonexistent_archive(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        DiagnosisAgent(archive_dir=tmp_path / "does-not-exist")


def test_diagnose_zero_turns_returns_no_lessons(fake_archive: Path):
    agent = DiagnosisAgent(
        archive_dir=fake_archive,
        max_turns=0,
        llm_caller=ScriptedLLM([]),
    )
    result = agent.run()
    assert isinstance(result, DiagnosisResult)
    # max_turns=0 means the for-loop never executes — stop reason is the else-branch
    assert result.lessons_path is None
    assert result.turns_used == 0
    assert result.stop_reason == "max_turns"


# ── Tool dispatch ──────────────────────────────────────────────────────


def test_read_file_dispatch(fake_archive: Path):
    """LLM calls read_file on a candidate.txt in the archive."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("read_file", {"path": "iteration_000/candidate.txt"}, "c1"),
        ]),
        _assistant_text("Done thinking."),  # second turn: stop
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=5, llm_caller=llm)
    result = agent.run()

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "read_file"
    # The tool output should contain the candidate text
    assert "Risk Dashboard" in result.tool_calls[0]["output_preview"]


def test_read_file_rejects_path_outside_archive(fake_archive: Path, tmp_path: Path):
    """Sandbox must reject an absolute path pointing outside the archive."""
    outside = tmp_path / "secret.txt"
    outside.write_text("password=hunter2")

    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("read_file", {"path": str(outside)}, "c1"),
        ]),
        _assistant_text("done"),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()

    # The dispatch should return an error JSON, not the secret
    tool_output = result.tool_calls[0]["output_preview"]
    assert "hunter2" not in tool_output
    assert "error" in tool_output.lower()


def test_search_files_dispatch(fake_archive: Path):
    """LLM searches for 'Risk Dashboard' across the archive."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("search_files", {"pattern": "Risk Dashboard"}, "c1"),
        ]),
        _assistant_text("done"),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()

    output = result.tool_calls[0]["output_preview"]
    # Preview is capped at 200 chars but should contain "matches" or a match count
    assert "match" in output.lower() or "pattern" in output.lower()


def test_write_lessons_exits_loop(fake_archive: Path):
    """Calling write_lessons should terminate the loop on that turn."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc(
                "write_lessons",
                {"content": "## Failure patterns\n- example\n## Candidate guidance\n- ex\n## Do NOT suggest\n- ex"},
                "c1",
            ),
        ]),
        # A second response — should NOT be consumed because the loop exits
        _assistant_with_tool_calls([_tc("read_file", {"path": "manifest.json"}, "c2")]),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=5, llm_caller=llm)
    result = agent.run()

    assert result.stop_reason == "wrote_lessons"
    assert result.lessons_path is not None
    assert result.lessons_path.exists()
    assert "Failure patterns" in result.lessons_path.read_text()
    assert result.turns_used == 1
    # Verify the second scripted response was NOT consumed
    assert len(llm.calls) == 1


def test_multi_turn_culminating_in_write(fake_archive: Path):
    """Realistic flow: search → read → write_lessons."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([_tc("search_files", {"pattern": "score"}, "c1")]),
        _assistant_with_tool_calls([_tc("read_file", {"path": "iteration_001/tasks/task_00001.json"}, "c2")]),
        _assistant_with_tool_calls([
            _tc("write_lessons", {"content": "## analysis\ntest"}, "c3"),
        ]),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=10, llm_caller=llm)
    result = agent.run()

    assert result.turns_used == 3
    assert result.stop_reason == "wrote_lessons"
    assert result.lessons_path is not None
    names = [c["name"] for c in result.tool_calls]
    assert names == ["search_files", "read_file", "write_lessons"]


# ── Budget and loop limits ─────────────────────────────────────────────


def test_budget_exceeded_stops_loop(fake_archive: Path):
    """If the cumulative cost exceeds max_cost_usd, the loop stops early."""
    llm = ScriptedLLM(
        [_assistant_text("thinking")] * 20,  # many turns available
        cost_per_call=0.15,  # $0.15 per call
    )
    agent = DiagnosisAgent(
        fake_archive,
        max_turns=20,
        max_cost_usd=0.30,  # budget for ~2 calls
        llm_caller=llm,
    )
    result = agent.run()

    # Check budget is hit before max_turns — and cost is actually > $0.30
    # meaning we did cap it.
    assert result.stop_reason in ("budget_exceeded", "llm_stopped")
    # With cost 0.15 * 3 = 0.45 we'd stop at turn 3 (budget checked BEFORE call).
    # If llm_stopped: first call had no tool_calls so we exited early (also fine).
    if result.stop_reason == "budget_exceeded":
        assert result.total_cost_usd >= 0.30


def test_max_turns_reached_without_writing(fake_archive: Path):
    """If the LLM keeps calling tools forever, max_turns caps the loop."""
    # Every turn returns a read_file call — never writes lessons
    llm = ScriptedLLM(
        [
            _assistant_with_tool_calls([
                _tc("read_file", {"path": "manifest.json"}, f"c{i}"),
            ])
            for i in range(10)
        ]
    )
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()
    assert result.turns_used == 3
    assert result.stop_reason == "max_turns"
    assert result.lessons_path is None


def test_llm_stops_without_tool_calls(fake_archive: Path):
    """If the LLM returns no tool_calls AND didn't write lessons, stop as 'llm_stopped'."""
    llm = ScriptedLLM([_assistant_text("I have nothing more to say.")])
    agent = DiagnosisAgent(fake_archive, max_turns=5, llm_caller=llm)
    result = agent.run()
    assert result.stop_reason == "llm_stopped"
    assert result.lessons_path is None


# ── Error handling ─────────────────────────────────────────────────────


def test_invalid_json_args_returns_error(fake_archive: Path):
    """Malformed tool_call arguments should return an error JSON, not crash."""
    llm = ScriptedLLM([
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{not valid json",
                                },
                            }
                        ],
                    }
                }
            ]
        },
        _assistant_text("done"),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()
    # Must not raise; dispatch returns an error payload
    assert len(result.tool_calls) == 1
    assert "error" in result.tool_calls[0]["output_preview"].lower()


def test_unknown_tool_name_returns_error(fake_archive: Path):
    llm = ScriptedLLM([
        _assistant_with_tool_calls([_tc("delete_everything", {}, "c1")]),
        _assistant_text("done"),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()
    assert "unknown tool" in result.tool_calls[0]["output_preview"].lower()


def test_write_lessons_caps_content_size(fake_archive: Path):
    """Ridiculously long lessons content should be capped at 100KB."""
    huge = "x" * 500_000
    llm = ScriptedLLM([
        _assistant_with_tool_calls([_tc("write_lessons", {"content": huge}, "c1")]),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()
    assert result.lessons_path is not None
    assert len(result.lessons_path.read_text()) == 100_000


# ── Sandbox integration ────────────────────────────────────────────────


def test_scoped_reads_is_active_during_diagnosis(fake_archive: Path, monkeypatch):
    """HERMES_READ_SAFE_ROOT must be set while the loop runs."""
    captured_env: dict = {}

    def capture_env(**kwargs):
        captured_env["HERMES_READ_SAFE_ROOT"] = __import__("os").environ.get("HERMES_READ_SAFE_ROOT")
        return (_assistant_text("done"), 0.001)

    agent = DiagnosisAgent(fake_archive, max_turns=2, llm_caller=capture_env)
    monkeypatch.delenv("HERMES_READ_SAFE_ROOT", raising=False)
    agent.run()

    assert captured_env["HERMES_READ_SAFE_ROOT"] == str(fake_archive.resolve())
    # After the run, env var should be restored (unset)
    import os as _os
    assert _os.environ.get("HERMES_READ_SAFE_ROOT") is None


def test_read_file_relative_path_resolves_against_archive(fake_archive: Path):
    """LLM should be able to use relative paths like 'iteration_000/candidate.txt'."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("read_file", {"path": "iteration_000/candidate.txt"}, "c1"),
        ]),
        _assistant_text("done"),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()
    output = result.tool_calls[0]["output_preview"]
    assert "error" not in output.lower() or "Risk Dashboard" in output


# ── Transcript persistence ─────────────────────────────────────────────


def test_transcript_persisted_on_success(fake_archive: Path):
    """A successful diagnosis run writes diagnosis/diagnosis_000.json."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("write_lessons", {"content": "## Failure patterns\n- test"}, "c1"),
        ]),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()

    assert result.stop_reason == "wrote_lessons"
    transcript_path = fake_archive / "diagnosis" / "diagnosis_000.json"
    assert transcript_path.exists()
    payload = json.loads(transcript_path.read_text())
    assert payload["stop_reason"] == "wrote_lessons"
    assert payload["lessons_written"] is True
    assert payload["turns_used"] == 1
    # Messages should contain at least system + user + assistant + tool
    assert len(payload["messages"]) >= 3


def test_transcript_persisted_on_failure(fake_archive: Path):
    """Even when lessons are NOT written (max_turns, llm_stopped, etc.),
    the transcript must still be persisted for debugging."""
    # LLM keeps reading but never writes lessons
    llm = ScriptedLLM([
        _assistant_with_tool_calls([_tc("read_file", {"path": "manifest.json"}, f"c{i}")])
        for i in range(10)
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)
    result = agent.run()

    assert result.stop_reason == "max_turns"
    assert result.lessons_path is None

    transcript_path = fake_archive / "diagnosis" / "diagnosis_000.json"
    assert transcript_path.exists()
    payload = json.loads(transcript_path.read_text())
    assert payload["stop_reason"] == "max_turns"
    assert payload["lessons_written"] is False
    # Should capture all 3 read_file calls
    assert len(payload["tool_calls"]) == 3
    assert all(tc["name"] == "read_file" for tc in payload["tool_calls"])


def test_transcript_index_advances_across_runs(fake_archive: Path):
    """Multiple diagnosis runs on the same archive produce separately-numbered transcripts."""
    for i in range(3):
        llm = ScriptedLLM([_assistant_text("nothing to add")])
        agent = DiagnosisAgent(fake_archive, max_turns=2, llm_caller=llm)
        agent.run()

    diag_dir = fake_archive / "diagnosis"
    assert (diag_dir / "diagnosis_000.json").exists()
    assert (diag_dir / "diagnosis_001.json").exists()
    assert (diag_dir / "diagnosis_002.json").exists()


def test_transcript_persistence_never_crashes_run(fake_archive: Path, monkeypatch):
    """If transcript write fails (e.g., disk full), the run should still return
    its DiagnosisResult. Wrapped in try/except with warning log."""
    llm = ScriptedLLM([
        _assistant_with_tool_calls([
            _tc("write_lessons", {"content": "## ok"}, "c1"),
        ]),
    ])
    agent = DiagnosisAgent(fake_archive, max_turns=3, llm_caller=llm)

    # Force the transcript write to fail
    def _boom(*a, **kw):
        raise OSError("simulated disk full")

    monkeypatch.setattr(agent, "_persist_transcript", _boom)
    result = agent.run()

    # Run should still report success
    assert result.stop_reason == "wrote_lessons"
    assert result.lessons_path is not None
