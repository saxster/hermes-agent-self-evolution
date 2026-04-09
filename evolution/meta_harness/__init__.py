"""Meta-harness extensions for hermes-agent-self-evolution.

Implements the filesystem-proposer insight from the Meta-Harness paper
(arXiv 2603.28052, Lee et al., Stanford/Anthropic):

    Rather than compressing optimization history into a short textual
    summary for the GEPA proposer, persist every candidate's source,
    per-task traces, and scores to disk as an archive. A diagnosis
    agent then reads the archive with grep/cat/ls and writes actionable
    lessons for the next iteration.

Phase A exports the TraceWriter plumbing. Later phases add the
path-scoped read tools (Phase B), the diagnosis agent (Phase C), and
the lessons-aware DSPy modules (Phase D).
"""

from evolution.meta_harness.trace_writer import (
    TraceWriter,
    get_active_writer,
    set_active_writer,
    make_tracing_metric,
    get_active_lessons,
    set_active_lessons,
    load_lessons_from_path,
)
from evolution.meta_harness.scoped_tools import scoped_reads, current_scope
from evolution.meta_harness.diagnose import DiagnosisAgent, DiagnosisResult

__all__ = [
    "TraceWriter",
    "get_active_writer",
    "set_active_writer",
    "make_tracing_metric",
    "get_active_lessons",
    "set_active_lessons",
    "load_lessons_from_path",
    "scoped_reads",
    "current_scope",
    "DiagnosisAgent",
    "DiagnosisResult",
]
