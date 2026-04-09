"""Per-task trace writer for GEPA optimization runs.

The TraceWriter persists everything a diagnosis agent will later need
to do counterfactual failure analysis:

    archive_dir/
    ├── iteration_000/
    │   ├── candidate.txt       # snapshot of candidate text at this iteration
    │   └── tasks/
    │       ├── task_00001.json # {task_input, expected, output, score, feedback}
    │       ├── task_00002.json
    │       └── ...
    ├── iteration_001/
    └── ...

DSPy's GEPA calls the fitness metric once per (example, candidate)
pair. The metric function receives the example and the prediction but
NOT the candidate text that produced the prediction — that lives on
the module instance GEPA is currently evaluating. We bridge this gap
with a thread-local "active writer" plus a set_candidate() call from
inside DSPy module.forward().

Design choices:
- Opt-in via HERMES_EVOLUTION_TRACING=1 env var. Default off: zero
  behavior change for existing code paths.
- Thread-local active writer so concurrent evolve_prompt and
  evolve_tool_desc runs don't collide.
- Atomic per-task JSON writes (rename from tmp) so a crash mid-run
  leaves a consistent archive for the diagnosis agent.
- Trace bodies capped at max_task_bytes (default 50 KB) to prevent
  a pathologically verbose agent_output from blowing up disk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Phase D: module-level "active lessons" that the DSPy proposer reads
# during forward() to get filesystem-derived guidance. Written by the
# Phase C diagnosis agent between iterations, loaded by a small helper.
_lessons_lock = threading.Lock()
_active_lessons: str = ""

logger = logging.getLogger(__name__)

# Environment variable that gates all tracing. Default off.
_ENV_ENABLED = "HERMES_EVOLUTION_TRACING"

# Per-task JSON body cap. Large agent_output strings get truncated with
# a marker. 50 KB × ~50 tasks × ~10 iterations ≈ 25 MB per run — tiny.
_DEFAULT_MAX_TASK_BYTES = 50_000

# Module-level global for the active writer. Used to be thread-local,
# but DSPy optimizers (GEPA, MIPROv2) run evaluation in worker threads
# via dspy.utils.parallelizer.ParallelExecutor — thread-local state
# does not propagate to those workers, which caused set_candidate()
# calls from inside module.forward() to silently become no-ops.
#
# A process-global writer means concurrent evolve_prompt + evolve_tool_desc
# runs in the SAME Python process would collide. Not a real use case
# for the CLI (each invocation is its own subprocess), and the module
# docstring warns against it.
_writer_lock = threading.Lock()
_active_writer_ref: Optional["TraceWriter"] = None


def tracing_enabled() -> bool:
    """Return True if tracing is opted in via the env var."""
    return os.environ.get(_ENV_ENABLED, "").strip().lower() in ("1", "true", "yes", "on")


class TraceWriter:
    """Persist per-task evaluation traces to an archive directory.

    Usage:
        writer = TraceWriter(Path("output/prompts/X/20260409_121314/traces"))
        set_active_writer(writer)
        try:
            # inside DSPy module.forward(), before returning the prediction:
            writer.set_candidate(self.section_text)
            # inside the wrapped metric function, after scoring:
            writer.log_eval(example, prediction, score, feedback="")
        finally:
            set_active_writer(None)
    """

    def __init__(
        self,
        archive_dir: Path | str,
        *,
        max_task_bytes: int = _DEFAULT_MAX_TASK_BYTES,
        artifact_name: str = "",
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.max_task_bytes = max_task_bytes
        self.artifact_name = artifact_name

        # State that changes as GEPA progresses.
        self._lock = threading.Lock()
        self._iteration = 0
        self._task_counter = 0
        self._current_candidate: str = ""
        self._candidates_seen: set[str] = set()  # hashes we've already snapshotted

        # Write a run-level manifest so the diagnosis agent can orient.
        manifest = {
            "schema_version": 1,
            "artifact_name": artifact_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "max_task_bytes": max_task_bytes,
        }
        manifest_path = self.archive_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ── Candidate tracking ─────────────────────────────────────────────

    def set_candidate(self, candidate_text: str) -> None:
        """Record the candidate text that will produce the next prediction.

        Called from inside a DSPy module's forward() just before it
        invokes its predictor. Advances the iteration counter when the
        candidate changes (i.e. GEPA proposed a new variant).
        """
        if candidate_text is None:
            candidate_text = ""
        candidate_text = str(candidate_text)
        with self._lock:
            text_hash = _hash(candidate_text)
            if text_hash not in self._candidates_seen:
                # New candidate — advance iteration and snapshot.
                if self._candidates_seen:
                    self._iteration += 1
                self._candidates_seen.add(text_hash)
                self._task_counter = 0
                self._current_candidate = candidate_text
                self._snapshot_candidate_locked()
            else:
                # Same candidate as the previous call.
                self._current_candidate = candidate_text

    def _snapshot_candidate_locked(self) -> None:
        """Write the current candidate text to iteration_NNN/candidate.txt."""
        iter_dir = self._iteration_dir_locked()
        iter_dir.mkdir(parents=True, exist_ok=True)
        path = iter_dir / "candidate.txt"
        try:
            _atomic_write_text(path, self._current_candidate)
        except OSError as exc:
            logger.warning("TraceWriter: candidate snapshot failed: %s", exc)

    # ── Per-task logging ───────────────────────────────────────────────

    def log_eval(
        self,
        example: Any,
        prediction: Any,
        score: float,
        *,
        feedback: str = "",
        extra: Optional[dict] = None,
    ) -> Optional[Path]:
        """Write a per-task trace and return the path, or None on failure."""
        try:
            with self._lock:
                self._task_counter += 1
                task_id = f"task_{self._task_counter:05d}"
                iteration = self._iteration
                candidate_preview = self._current_candidate[:200]

            task_input = _get(example, "task_input", "")
            expected = _get(example, "expected_behavior", "")
            category = _get(example, "category", "")
            difficulty = _get(example, "difficulty", "")
            agent_output = _get(prediction, "output", "")

            trace: dict[str, Any] = {
                "task_id": task_id,
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_input": _truncate(task_input, self.max_task_bytes // 4),
                "expected_behavior": _truncate(expected, self.max_task_bytes // 4),
                "agent_output": _truncate(agent_output, self.max_task_bytes // 2),
                "score": float(score),
                "feedback": _truncate(feedback, self.max_task_bytes // 8),
                "category": category,
                "difficulty": difficulty,
                "candidate_preview": candidate_preview,
                "candidate_chars": len(self._current_candidate),
            }
            if extra:
                trace["extra"] = extra

            iter_dir = self._iteration_dir(iteration)
            tasks_dir = iter_dir / "tasks"
            tasks_dir.mkdir(parents=True, exist_ok=True)
            path = tasks_dir / f"{task_id}.json"
            payload = json.dumps(trace, indent=2, ensure_ascii=False)
            _atomic_write_text(path, payload)
            return path
        except Exception as exc:  # noqa: BLE001 — tracing must never crash GEPA
            logger.warning("TraceWriter: log_eval failed: %s", exc)
            return None

    # ── Introspection / path helpers ───────────────────────────────────

    def _iteration_dir_locked(self) -> Path:
        return self.archive_dir / f"iteration_{self._iteration:03d}"

    def _iteration_dir(self, iteration: int) -> Path:
        return self.archive_dir / f"iteration_{iteration:03d}"

    @property
    def iteration(self) -> int:
        with self._lock:
            return self._iteration

    @property
    def task_count(self) -> int:
        with self._lock:
            return self._task_counter


# ── Active writer (process-global) ─────────────────────────────────────


def get_active_writer() -> Optional["TraceWriter"]:
    """Return the currently-active writer, or None."""
    with _writer_lock:
        return _active_writer_ref


def set_active_writer(writer: Optional["TraceWriter"]) -> None:
    """Install (or clear) the active writer.

    Process-global by design — DSPy's ParallelExecutor runs module.forward()
    in worker threads that don't see thread-local state. A global with a
    lock lets the forward() hook reach the writer from any thread.
    """
    global _active_writer_ref
    with _writer_lock:
        _active_writer_ref = writer


# ── Metric wrapping ────────────────────────────────────────────────────


def make_tracing_metric(
    inner_metric: Callable[..., float],
    on_iteration_complete: Optional[Callable[[int], None]] = None,
) -> Callable[..., float]:
    """Wrap a DSPy-compatible fitness metric so each call logs a trace.

    The wrapped function has the same signature as the input metric
    ``(example, prediction, trace=None) -> float`` and still returns
    the inner metric's score unchanged. If tracing is disabled or no
    writer is active, this is a no-op passthrough with negligible cost.

    If ``on_iteration_complete`` is provided, it is called with the
    just-finished iteration number whenever the writer's internal
    iteration counter advances. This is the hook Phase C uses to run
    the diagnosis agent between GEPA iterations.
    """

    _state = {"last_iteration_seen": 0}

    def wrapped(example, prediction, trace=None, *args, **kwargs):
        score = inner_metric(example, prediction, trace, *args, **kwargs)
        if not tracing_enabled():
            return score
        writer = get_active_writer()
        if writer is None:
            return score

        writer.log_eval(example, prediction, score)

        if on_iteration_complete is not None:
            current = writer.iteration
            if current > _state["last_iteration_seen"]:
                finished = _state["last_iteration_seen"]
                _state["last_iteration_seen"] = current
                try:
                    on_iteration_complete(finished)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "on_iteration_complete callback failed for iteration %d: %s",
                        finished,
                        exc,
                    )
        return score

    wrapped.__wrapped__ = inner_metric  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(inner_metric, "__name__", "tracing_metric")
    return wrapped


# ── Lessons (Phase D) ──────────────────────────────────────────────────


def set_active_lessons(lessons: str) -> None:
    """Install the lessons text that DSPy modules will read in forward()."""
    global _active_lessons
    with _lessons_lock:
        _active_lessons = lessons or ""


def get_active_lessons() -> str:
    """Return the current active lessons text, or empty string if none."""
    with _lessons_lock:
        return _active_lessons


def load_lessons_from_path(path: Path | str) -> str:
    """Read a lessons.md file and install it as active lessons.

    Returns the loaded text (empty string if the file doesn't exist).
    """
    p = Path(path)
    try:
        if p.exists():
            text = p.read_text(encoding="utf-8")
        else:
            text = ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_lessons_from_path: %s: %s", p, exc)
        text = ""
    set_active_lessons(text)
    return text


# ── Internal helpers ───────────────────────────────────────────────────


def _get(obj: Any, name: str, default: Any = "") -> Any:
    """Safely pull an attribute from a dspy.Example or dspy.Prediction."""
    try:
        val = getattr(obj, name, None)
        if val is None and hasattr(obj, "__getitem__"):
            try:
                val = obj[name]
            except (KeyError, TypeError, IndexError):
                val = None
        return val if val is not None else default
    except Exception:  # noqa: BLE001
        return default


def _truncate(value: Any, max_bytes: int) -> str:
    """Clamp a string to max_bytes. Non-string inputs are stringified."""
    s = value if isinstance(value, str) else str(value)
    if max_bytes <= 0:
        return s
    encoded = s.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    suffix = f"\n... [truncated {len(encoded) - max_bytes} bytes]"
    return truncated + suffix


def _hash(text: str) -> str:
    """Short stable hash for candidate deduplication (not security)."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via a tmp file + rename."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
