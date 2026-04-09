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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evolution.meta_harness.scoped_tools import scoped_reads

logger = logging.getLogger(__name__)

# Hard cap on the content returned by a single read_file or search_files
# call. Prevents a single huge trace from blowing the diagnosis LLM's
# context window. 40 KB is roughly 10-15K tokens — safe for any model.
_MAX_TOOL_OUTPUT_CHARS = 40_000

# Diagnosis system prompt — REVISED 2026-04-09 after observing that
# gpt-4.1-mini wasted 10-15 turns on exploration and never reached
# write_lessons. New version embeds a pre-computed archive summary
# in the user message so the LLM has all data upfront, and demands
# that write_lessons be called on turn 1 or 2.
_DIAGNOSIS_SYSTEM_PROMPT = """You are a diagnostic agent. You will receive a pre-computed summary of an AI agent's optimization archive. Your single job is to write a lessons.md file that helps the NEXT optimizer iteration avoid repeating mistakes.

CRITICAL RULES:
1. You MUST call `write_lessons` on your first response. Do NOT explore the archive first — the summary you receive already contains the data you need.
2. If you want more detail on a specific trace, you may call `read_file` ONCE before writing lessons — but only if the summary is missing a concrete quote you need.
3. Do NOT call `search_files` — the summary already has score distributions and category breakdowns.
4. lessons.md must have exactly THREE sections:

   ## Failure patterns
   3–5 SPECIFIC observations. Reference iteration numbers and scores from the summary.
   GOOD: "Iteration 002 scored 0.91; iteration 005 dropped the 'Risk Dashboard' section and scored 0.62."
   BAD: "Agent should be more detailed."

   ## Candidate guidance
   Concrete suggestions for the NEXT candidate. Quote exact phrases from the highest-scoring candidate.

   ## Do NOT suggest
   Things already tried that hurt scores. Name iterations and score deltas.

If the archive has fewer than 2 iterations, write a short lessons.md saying "Not enough data yet — iterate more." That is still a valid output.

Call write_lessons ONCE. The loop will end when you do."""


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
        """Execute the diagnosis loop. Returns a DiagnosisResult.

        Side effect: writes a diagnosis transcript to
        ``{archive_dir}/diagnosis/diagnosis_{N}.json`` where N is the
        next sequential index, so repeat invocations don't overwrite.
        The transcript contains the full message history + tool call
        detail + cost + stop reason. Persisted EVEN when lessons.md is
        not written — that's the whole point (debug failure modes).
        """
        result = DiagnosisResult()
        # Pre-compute an archive summary so the LLM doesn't need to
        # waste turns on exploration. This was the #1 failure mode in
        # the first real-LLM A/B run — gpt-4.1-mini burned 10-15 turns
        # on search_files before even attempting write_lessons.
        archive_summary = self._build_archive_summary()
        messages = [
            {
                "role": "system",
                "content": _DIAGNOSIS_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"# Archive summary\n\n{archive_summary}\n\n"
                    f"# Your task\n\n"
                    f"Based on the summary above, call `write_lessons` with a "
                    f"markdown document following the three-section format. "
                    f"Do not explore the archive unless you need one specific "
                    f"quote. Budget: {self.max_turns} turns, "
                    f"${self.max_cost_usd:.2f}."
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

        # Persist the full transcript ALWAYS — whether lessons.md was
        # written or not. This is load-bearing for debugging diagnosis
        # failures without re-running (which costs real LLM money).
        try:
            self._persist_transcript(messages, result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcript persistence failed: %s", exc)

        return result

    def _build_archive_summary(self) -> str:
        """Walk the archive and build a compact text summary for the LLM.

        Format (deterministic, ~1-3 KB per iteration):

            ## manifest
            {key fields from manifest.json}

            ## iteration_000
            candidate (first 600 chars): ...
            scores: n=N, min=X, max=Y, mean=Z
            tasks:
              - task_00001 (score=0.42): task_input="..."
              - task_00002 (score=0.91): task_input="..."
            lowest 2:
              task_00003 score=0.12 — agent_output preview: "..."
              task_00004 score=0.18 — agent_output preview: "..."

        Each iteration is limited so even 20 iterations × 50 tasks fit
        in a reasonable prompt budget (~40-80 KB total).
        """
        lines: list[str] = []

        # Manifest
        manifest_path = self.archive_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                lines.append("## manifest")
                for k in ("artifact_name", "created_at", "schema_version"):
                    if k in manifest:
                        lines.append(f"  {k}: {manifest[k]}")
                lines.append("")
            except Exception:  # noqa: BLE001
                pass

        # Iterations
        iter_dirs = sorted(
            p for p in self.archive_dir.iterdir()
            if p.is_dir() and p.name.startswith("iteration_")
        )
        if not iter_dirs:
            lines.append("(No iterations present yet — archive is empty.)")
            return "\n".join(lines)

        for iter_dir in iter_dirs:
            lines.append(f"## {iter_dir.name}")

            # Candidate
            candidate_path = iter_dir / "candidate.txt"
            if candidate_path.exists():
                try:
                    ctext = candidate_path.read_text(encoding="utf-8", errors="replace")
                    # Cap at 600 chars so the composite doesn't dominate
                    preview = ctext[:600]
                    if len(ctext) > 600:
                        preview += f"... [+{len(ctext) - 600} chars]"
                    # Indent so markdown renders it as a block
                    indented = "\n".join("    " + line for line in preview.split("\n"))
                    lines.append("candidate:")
                    lines.append(indented)
                except Exception:  # noqa: BLE001
                    lines.append("candidate: (unreadable)")

            # Task score stats
            tasks_dir = iter_dir / "tasks"
            task_files = sorted(tasks_dir.glob("*.json")) if tasks_dir.exists() else []
            if not task_files:
                lines.append("tasks: (none)")
                lines.append("")
                continue

            scored: list[tuple[float, str, dict]] = []  # (score, task_id, trace)
            for tf in task_files:
                try:
                    trace = json.loads(tf.read_text(encoding="utf-8"))
                    score = float(trace.get("score", 0.0) or 0.0)
                    scored.append((score, tf.stem, trace))
                except Exception:  # noqa: BLE001
                    continue

            if scored:
                scores_only = [s for s, _, _ in scored]
                n = len(scores_only)
                mean = sum(scores_only) / n
                lines.append(f"scores: n={n}, min={min(scores_only):.3f}, max={max(scores_only):.3f}, mean={mean:.3f}")

                # One-line per task with score + task_input preview
                lines.append("tasks:")
                for sc, tid, trace in scored:
                    ti = str(trace.get("task_input", ""))[:100]
                    lines.append(f'  - {tid} (score={sc:.3f}): "{ti}"')

                # Lowest 2 with agent_output preview so the LLM can
                # see WHY they failed without calling read_file
                lowest = sorted(scored, key=lambda t: t[0])[:2]
                if lowest:
                    lines.append("lowest scoring (with agent_output preview):")
                    for sc, tid, trace in lowest:
                        ao = str(trace.get("agent_output", ""))[:300]
                        lines.append(f'  {tid} (score={sc:.3f})')
                        lines.append(f'    expected: {str(trace.get("expected_behavior", ""))[:150]}')
                        lines.append(f'    actual:   {ao}')
            lines.append("")

        return "\n".join(lines)

    def _persist_transcript(self, messages: list[dict], result: DiagnosisResult) -> None:
        """Write a full diagnosis transcript to the archive for debugging.

        File name: diagnosis/diagnosis_{N:03d}.json, with N incremented
        sequentially so repeat invocations don't overwrite prior runs.
        """
        diag_dir = self.archive_dir / "diagnosis"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Find the next available index
        existing = sorted(diag_dir.glob("diagnosis_*.json"))
        next_idx = len(existing)
        path = diag_dir / f"diagnosis_{next_idx:03d}.json"

        payload = {
            "archive_dir": str(self.archive_dir),
            "model_name": self.model_name,
            "max_turns": self.max_turns,
            "max_cost_usd": self.max_cost_usd,
            "temperature": self.temperature,
            "stop_reason": result.stop_reason,
            "turns_used": result.turns_used,
            "total_cost_usd": result.total_cost_usd,
            "error": result.error,
            "lessons_written": result.lessons_path is not None,
            "tool_calls": result.tool_calls,  # name + args_preview + output_preview
            "messages": messages,  # full conversation including system + tool results
            "persisted_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

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
