"""Path-scoped file-tool helpers for the meta-harness diagnosis agent.

The diagnosis agent (Phase C) needs read_file + search_files capability
limited to a specific archive directory — never SOUL.md, ~/.hermes/.env,
session databases, memories, or other sensitive locations.

We implement this by setting the HERMES_READ_SAFE_ROOT environment
variable around the diagnosis run. The read-safe-root check is
implemented inside `hermes-agent/tools/file_operations.py` (the
`_is_read_denied()` function) and enforced by both read_file_tool and
search_tool in `hermes-agent/tools/file_tools.py`.

Usage:
    from evolution.meta_harness.scoped_tools import scoped_reads

    with scoped_reads(Path("output/prompts/X/20260409_*/traces")):
        # inside this block, any hermes file tool read is confined to
        # the archive. Attempts to read anything else return an error.
        diagnosis_agent.run()

    # outside the block, the original env var value (if any) is restored.

Notes:
- This is a thin wrapper; the real enforcement is the env var check.
- Nested use is supported: inner blocks temporarily override the outer
  root, and restore it on exit.
- The context manager is thread-unsafe by design (env vars are
  process-global). Do not use across concurrent diagnosis agents.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_ENV_VAR = "HERMES_READ_SAFE_ROOT"


@contextmanager
def scoped_reads(archive_dir: Path | str) -> Iterator[Path]:
    """Context manager that sets HERMES_READ_SAFE_ROOT for the block.

    Yields the resolved archive path. Restores the previous env var
    value (or unsets it if it wasn't set) on exit, even on exceptions.
    """
    resolved = Path(archive_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"scoped_reads: archive does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"scoped_reads: archive is not a directory: {resolved}")

    previous = os.environ.get(_ENV_VAR)
    os.environ[_ENV_VAR] = str(resolved)
    try:
        yield resolved
    finally:
        if previous is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = previous


def current_scope() -> Path | None:
    """Return the currently-active scope path, or None if unset."""
    val = os.environ.get(_ENV_VAR, "")
    if not val:
        return None
    try:
        return Path(val).resolve()
    except Exception:  # noqa: BLE001
        return None
