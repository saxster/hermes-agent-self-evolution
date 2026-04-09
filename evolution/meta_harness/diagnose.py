"""Diagnosis agent — the actual filesystem-proposer from the Meta-Harness paper.

This is Phase C of the filesystem-proposer bolt-on. Between GEPA iterations,
we spawn a small LLM agent with THREE tools scoped to a specific archive:

    read_file(path, offset=1, limit=500)
        Read a file from the archive. Returns line-numbered content.
    search_files(pattern, path=".", file_glob=None)
        Regex search inside archive files. ripgrep-style.
    write_lessons(content)
        Write lessons.md at the archive root. The ONLY writable path.

The agent uses these tools to do counterfactual failure diagnosis:
    - Which task categories fail repeatedly?
    - What's different between high-scoring and low-scoring candidates?
    - What has already been tried and should NOT be suggested again?

It then writes a lessons.md file that Phase D will inject into the GEPA
proposer's input on subsequent iterations.

Safety:
    - The read tools honor HERMES_READ_SAFE_ROOT (set by scoped_reads())
      so the agent cannot read SOUL.md, .env files, session DBs, or
      anything else outside the archive.
    - The write tool is a FIXED-PATH helper that only writes to
      {archive_dir}/lessons.md — the LLM does not choose the path.
    - The loop has a turn cap AND a hard cost cap (USD). Whichever
      triggers first stops the loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from evolution.meta_harness.scoped_tools import scoped_reads

logger = logging.getLogger(__name__)

# Hard cap on the content returned by a single read_file or search_files
# call. Prevents a single huge trace from blowing the diagnosis LLM's
# context window. 40 KB is roughly 10-15K tokens — safe for any model.
_MAX_TOOL_OUTPUT_CHARS = 40_000

# Default diagnosis prompt. Deliberately specific about what makes a
# "good" lessons.md vs a useless one, with a concrete example.
_DIAGNOSIS_SYSTEM_PROMPT = """You are a diagnostic agent analyzing an AI agent's optimization archive.

The archive at {archive_dir} contains traces from prior candidate variants of an AI agent's system prompt (or tool description). Structure:

    {archive_dir}/
    ├── manifest.json             — run metadata
    ├── iteration_000/
    │   ├── candidate.txt         — the candidate text at this iteration
    │   └── tasks/
    │       ├── task_00001.json   — {{task_input, expected, agent_output, score, feedback, category}}
    │       ├── task_00002.json
    │       └── ...
    ├── iteration_001/
    └── ...

Your job:
1. Use `search_files` to find patterns across iterations. Look at score distributions — which task categories consistently fail? Which candidates scored highest?
2. Use `read_file` to inspect specific failing traces in detail.
3. Compare high-scoring candidates vs low-scoring candidates. What's different in the candidate text? What did the successful agent_output do that the failing ones didn't?
4. Call `write_lessons` with a markdown document containing THREE sections:

   ## Failure patterns
   3–5 specific, actionable observations. Cite task_ids and iteration numbers.
   Example GOOD: "Tasks 003 and 007 both failed because the candidate omitted the 'Risk Dashboard' section. Iteration 002 included it and scored 0.91; iteration 005 dropped it and scored 0.62."
   Example BAD: "Agent should be more detailed."

   ## Candidate guidance
   Concrete suggestions for the NEXT candidate. Quote exact phrases from successful candidates.

   ## Do NOT suggest
   Things already tried that hurt scores. Name the iteration and the score delta.

Call write_lessons ONCE at the end with your full markdown. Do not call it more than once. When you are done, stop — the loop will end automatically.

Be specific. References like "iteration 003 task_00007" are far more useful than general advice."""


@dataclass
class DiagnosisResult:
    """Outcome of a diagnosis run."""

    lessons_path: Optional[Path] = None
    turns_used: int = 0
    total_cost_usd: float = 0.0
    stop_reason: str = ""  # "wrote_lessons" | "max_turns" | "budget_exceeded" | "llm_stopped" | "error"
    tool_calls: list[dict] = field(default_factory=list)  # for tests/debugging
    error: Optional[str] = None


class DiagnosisAgent:
    """Spawns a tool-using LLM agent scoped to an optimization archive.

    Usage:
        agent = DiagnosisAgent(
            archive_dir=Path("output/prompts/X/20260409_121314/traces"),
            model_name="openai/gpt-4.1-mini",
            max_turns=15,
            max_cost_usd=0.50,
        )
        result = agent.run()
        if result.lessons_path:
            print(f"Wrote {result.lessons_path}")
    """

    def __init__(
        self,
        archive_dir: Path | str,
        model_name: str = "openai/gpt-4.1-mini",
        *,
        max_turns: int = 15,
        max_cost_usd: float = 0.50,
        temperature: float = 0.2,
        llm_caller: Optional[Any] = None,
    ) -> None:
        self.archive_dir = Path(archive_dir).expanduser().resolve()
        if not self.archive_dir.is_dir():
            raise NotADirectoryError(f"archive_dir not found: {self.archive_dir}")

        self.model_name = model_name
        self.max_turns = max_turns
        self.max_cost_usd = max_cost_usd
        self.temperature = temperature

        # Injected LLM caller for tests. If None, we lazy-import litellm.
        # Signature: llm_caller(model, messages, tools, temperature) -> (response, cost)
        self._llm_caller = llm_caller

        self.lessons_path = self.archive_dir / "lessons.md"

    # ── Public entry point ─────────────────────────────────────────────

    def run(self) -> DiagnosisResult:
        """Execute the diagnosis loop. Returns a DiagnosisResult."""
        result = DiagnosisResult()
        messages = [
            {
                "role": "system",
                "content": _DIAGNOSIS_SYSTEM_PROMPT.format(archive_dir=str(self.archive_dir)),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the optimization archive at {self.archive_dir} and "
                    f"write a lessons.md file. You have at most {self.max_turns} "
                    f"turns and a ${self.max_cost_usd:.2f} budget."
                ),
            },
        ]
        tools = self._tool_schemas()

        try:
            with scoped_reads(self.archive_dir):
                for turn in range(self.max_turns):
                    result.turns_used = turn + 1

                    # Budget gate BEFORE the call, not after, so a runaway
                    # turn can't push us over.
                    if result.total_cost_usd >= self.max_cost_usd:
                        result.stop_reason = "budget_exceeded"
                        break

                    response, cost = self._call_llm(messages, tools)
                    result.total_cost_usd += cost

                    choice = response["choices"][0]
                    msg = choice["message"]

                    # Append the assistant turn to the conversation.
                    # Keep only the fields the API expects back.
                    assistant_msg: dict = {"role": "assistant"}
                    if msg.get("content") is not None:
                        assistant_msg["content"] = msg.get("content", "")
                    if msg.get("tool_calls"):
                        assistant_msg["tool_calls"] = msg["tool_calls"]
                    messages.append(assistant_msg)

                    tool_calls = msg.get("tool_calls") or []
                    if not tool_calls:
                        # LLM stopped calling tools. If lessons.md exists,
                        # this is success; otherwise the LLM gave up.
                        result.stop_reason = (
                            "wrote_lessons" if self.lessons_path.exists() else "llm_stopped"
                        )
                        break

                    # Dispatch every tool call in this turn.
                    wrote_lessons = False
                    for tc in tool_calls:
                        tool_output = self._dispatch(tc)
                        result.tool_calls.append(
                            {
                                "name": _tc_name(tc),
                                "args_preview": _tc_args(tc)[:500],
                                "output_preview": tool_output[:2000],
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_output,
                            }
                        )
                        if _tc_name(tc) == "write_lessons" and self.lessons_path.exists():
                            wrote_lessons = True

                    if wrote_lessons:
                        result.stop_reason = "wrote_lessons"
                        break
                else:
                    result.stop_reason = "max_turns"

        except Exception as exc:  # noqa: BLE001
            logger.exception("DiagnosisAgent failed")
            result.stop_reason = "error"
            result.error = str(exc)

        if self.lessons_path.exists():
            result.lessons_path = self.lessons_path
        return result

    # ── LLM call ───────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> tuple[dict, float]:
        """Invoke the LLM. Returns (response_dict, cost_usd)."""
        if self._llm_caller is not None:
            return self._llm_caller(
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=self.temperature,
            )

        # Default path: litellm.completion
        import litellm

        raw = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        # Normalize to plain dicts so tests and prod share one path.
        response_dict = (
            raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)  # type: ignore[arg-type]
        )
        try:
            cost = float(litellm.completion_cost(completion_response=raw) or 0.0)
        except Exception:  # noqa: BLE001
            cost = 0.0
        return response_dict, cost

    # ── Tool schemas (OpenAI format) ───────────────────────────────────

    def _tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a file from the optimization archive. "
                        "Returns the content with line numbers. "
                        f"Only paths inside {self.archive_dir} are allowed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to read (absolute or relative to archive root).",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "1-indexed line to start from (default 1).",
                                "default": 1,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max lines to return (default 500, cap 2000).",
                                "default": 500,
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": (
                        "Regex search inside archive files. Returns matching lines "
                        "with file:line prefix. Use this to find patterns across many "
                        "traces at once (e.g. all failing tasks, all iterations with a keyword)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern.",
                            },
                            "path": {
                                "type": "string",
                                "description": "Subdirectory inside the archive (default = archive root).",
                                "default": ".",
                            },
                            "file_glob": {
                                "type": "string",
                                "description": "Filter files by glob (e.g. '*.json').",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_lessons",
                    "description": (
                        "Write the final lessons.md file. Call this ONCE at the end "
                        "when your analysis is complete. The file is written at the "
                        "archive root; you do not choose the path."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "Full markdown content. Should include three "
                                    "sections: '## Failure patterns', '## Candidate "
                                    "guidance', '## Do NOT suggest'."
                                ),
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
        ]

    # ── Tool dispatch ──────────────────────────────────────────────────

    def _dispatch(self, tool_call: dict) -> str:
        name = _tc_name(tool_call)
        try:
            args = json.loads(_tc_args(tool_call) or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON args: {exc}"})

        if name == "read_file":
            return self._tool_read_file(**args)
        if name == "search_files":
            return self._tool_search_files(**args)
        if name == "write_lessons":
            return self._tool_write_lessons(**args)
        return json.dumps({"error": f"unknown tool: {name}"})

    def _tool_read_file(self, path: str, offset: int = 1, limit: int = 500) -> str:
        # Resolve relative paths against the archive root FIRST, then
        # run the sandbox check on the fully-resolved absolute path.
        # Checking the raw relative path would resolve it against CWD
        # (wrong) and reject legitimate in-archive paths.
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (self.archive_dir / p).resolve()
        else:
            p = p.resolve()

        denied, reason = _is_read_denied_safe(str(p))
        if denied:
            return json.dumps({"error": reason})

        try:
            if not p.exists():
                return json.dumps({"error": f"no such file: {p}"})
            if p.is_dir():
                entries = sorted(e.name for e in p.iterdir())[:200]
                return json.dumps({"entries": entries, "is_dir": True})

            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            # 1-indexed offset, cap limit
            start = max(1, int(offset)) - 1
            end = min(len(lines), start + min(int(limit), 2000))
            excerpt = "\n".join(f"{i + 1}|{line}" for i, line in enumerate(lines[start:end], start=start))
            if len(excerpt) > _MAX_TOOL_OUTPUT_CHARS:
                excerpt = excerpt[:_MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
            return json.dumps(
                {
                    "path": str(p),
                    "total_lines": len(lines),
                    "returned_lines": f"{start + 1}-{end}",
                    "content": excerpt,
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"read failed: {exc}"})

    def _tool_search_files(
        self, pattern: str, path: str = ".", file_glob: Optional[str] = None
    ) -> str:
        # Resolve search root relative to archive_dir.
        root = Path(path).expanduser()
        if not root.is_absolute():
            root = (self.archive_dir / root).resolve()
        denied, reason = _is_read_denied_safe(str(root))
        if denied:
            return json.dumps({"error": reason})
        if not root.exists():
            return json.dumps({"error": f"no such path: {root}"})

        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            return json.dumps({"error": f"invalid regex: {exc}"})

        glob_pattern = file_glob or "**/*"
        matches: list[dict] = []
        total_scanned = 0
        for entry in root.rglob(glob_pattern) if root.is_dir() else [root]:
            if not entry.is_file():
                continue
            total_scanned += 1
            # Don't try to read binary junk
            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "file": str(entry.relative_to(self.archive_dir)),
                            "line": lineno,
                            "text": line[:300],
                        }
                    )
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break

        payload = json.dumps(
            {
                "pattern": pattern,
                "path": str(root),
                "matches_found": len(matches),
                "files_scanned": total_scanned,
                "matches": matches,
            },
            ensure_ascii=False,
        )
        if len(payload) > _MAX_TOOL_OUTPUT_CHARS:
            payload = payload[:_MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
        return payload

    def _tool_write_lessons(self, content: str) -> str:
        # Fixed path — the LLM does not pick where this goes.
        try:
            # Cap lessons size so an infinite-generation bug can't fill disk.
            capped = content[:100_000]
            self.lessons_path.write_text(capped, encoding="utf-8")
            return json.dumps(
                {
                    "wrote": str(self.lessons_path),
                    "chars": len(capped),
                    "status": "lessons.md written — stop the loop now",
                }
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"write failed: {exc}"})


# ── Helpers ────────────────────────────────────────────────────────────


def _tc_name(tc: dict) -> str:
    fn = tc.get("function") or {}
    return fn.get("name") or ""


def _tc_args(tc: dict) -> str:
    fn = tc.get("function") or {}
    return fn.get("arguments") or "{}"


def _is_read_denied_safe(path: str) -> tuple[bool, Optional[str]]:
    """Call _is_read_denied via cross-repo import.

    Self-evolution does not import hermes-agent Python modules at module
    load time (to stay decoupled), so we do the import lazily inside the
    check function with a fallback.
    """
    try:
        import sys

        hermes_root = (
            Path(__file__).parent.parent.parent.parent / "hermes-agent"
        )
        if hermes_root.exists():
            path_str = str(hermes_root)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            from tools.file_operations import _is_read_denied  # type: ignore

            return _is_read_denied(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta-harness: cross-repo import failed, using local check: %s", exc)

    # Fallback: local reimplementation
    return _local_read_denied(path)


def _local_read_denied(path: str) -> tuple[bool, Optional[str]]:
    """Minimal fallback mirror of hermes-agent's _is_read_denied."""
    root = os.environ.get("HERMES_READ_SAFE_ROOT", "")
    if not root:
        return (False, None)
    try:
        safe_root = os.path.realpath(os.path.expanduser(root))
        resolved = os.path.realpath(os.path.expanduser(path))
    except Exception as exc:  # noqa: BLE001
        return (True, f"Cannot resolve path: {exc}")
    if resolved == safe_root or resolved.startswith(safe_root + os.sep):
        return (False, None)
    return (True, f"Read denied: '{path}' outside sandbox '{safe_root}'")
