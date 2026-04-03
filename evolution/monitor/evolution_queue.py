"""Persistent priority queue for evolution targets.

Stores queued evolution jobs in ~/.hermes/evolution/queue.json with:
- Priority ordering (lower number = higher priority)
- Deduplication (won't re-enqueue something already pending/in-progress)
- Cooldown tracking (won't re-trigger the same target within a configurable window)
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

QUEUE_PATH = Path.home() / ".hermes" / "evolution" / "queue.json"
DEFAULT_COOLDOWN_HOURS = 72


@dataclass
class QueueItem:
    """A single item in the evolution queue."""
    target_type: str  # "skill", "tool", "prompt"
    target_name: str  # e.g. "github-code-review" or "file_tools"
    priority: int  # Lower = higher priority (1 is highest)
    reason: str  # Why this was enqueued
    enqueued_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, in_progress, completed, failed
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_summary: Optional[str] = None

    @property
    def age_hours(self) -> float:
        return (time.time() - self.enqueued_at) / 3600

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QueueItem":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


class EvolutionQueue:
    """Priority queue with persistence and cooldown tracking.

    Queue state is persisted to ~/.hermes/evolution/queue.json so it
    survives process restarts.
    """

    def __init__(
        self,
        queue_path: Optional[Path] = None,
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
    ):
        self.queue_path = queue_path or QUEUE_PATH
        self.cooldown_hours = cooldown_hours
        self._items: list[QueueItem] = []
        self._load()

    def enqueue(
        self,
        target_type: str,
        target_name: str,
        priority: int = 5,
        reason: str = "",
    ) -> bool:
        """Add a target to the evolution queue.

        Returns True if enqueued, False if deduplicated or in cooldown.
        """
        # Dedup: skip if already pending or in progress
        if self.is_pending(target_name):
            return False

        # Cooldown: skip if completed recently
        if self._in_cooldown(target_name):
            return False

        item = QueueItem(
            target_type=target_type,
            target_name=target_name,
            priority=priority,
            reason=reason,
        )
        self._items.append(item)
        self._sort()
        self._save()
        return True

    def dequeue(self) -> Optional[QueueItem]:
        """Pop the highest-priority pending item and mark it in_progress.

        Returns None if the queue is empty.
        """
        for item in self._items:
            if item.status == "pending":
                item.status = "in_progress"
                item.started_at = time.time()
                self._save()
                return item
        return None

    def is_pending(self, target_name: str) -> bool:
        """Check if a target is already pending or in progress."""
        active_statuses = {"pending", "in_progress"}
        return any(
            item.target_name == target_name and item.status in active_statuses
            for item in self._items
        )

    def mark_complete(
        self,
        target_name: str,
        success: bool = True,
        summary: str = "",
    ) -> bool:
        """Mark an in-progress item as completed or failed.

        Returns True if the item was found and updated.
        """
        for item in self._items:
            is_match = item.target_name == target_name
            is_active = item.status == "in_progress"
            if is_match and is_active:
                item.status = "completed" if success else "failed"
                item.completed_at = time.time()
                item.result_summary = summary
                self._save()
                return True
        return False

    def get_pending(self) -> list[QueueItem]:
        """Get all pending items, sorted by priority."""
        return [item for item in self._items if item.status == "pending"]

    def get_all(self) -> list[QueueItem]:
        """Get all items regardless of status."""
        return list(self._items)

    def prune_completed(self, max_age_hours: float = 168) -> int:
        """Remove completed/failed items older than max_age_hours (default 7 days).

        Returns count of pruned items.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        terminal_statuses = {"completed", "failed"}
        before_count = len(self._items)

        self._items = [
            item for item in self._items
            if not (
                item.status in terminal_statuses
                and (item.completed_at or item.enqueued_at) < cutoff
            )
        ]

        pruned = before_count - len(self._items)
        if pruned > 0:
            self._save()
        return pruned

    def _in_cooldown(self, target_name: str) -> bool:
        """Check if a target was completed recently (within cooldown window)."""
        cutoff = time.time() - (self.cooldown_hours * 3600)
        return any(
            item.target_name == target_name
            and item.status in {"completed", "failed"}
            and (item.completed_at or 0) > cutoff
            for item in self._items
        )

    def _sort(self):
        """Sort items by priority (ascending), then by enqueue time."""
        self._items.sort(key=lambda x: (x.priority, x.enqueued_at))

    def _load(self):
        """Load queue state from disk."""
        if not self.queue_path.exists():
            self._items = []
            return

        try:
            raw = json.loads(self.queue_path.read_text())
            self._items = [QueueItem.from_dict(item) for item in raw]
            self._sort()
        except (json.JSONDecodeError, KeyError, TypeError):
            self._items = []

    def _save(self):
        """Persist queue state to disk."""
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [item.to_dict() for item in self._items]
        self.queue_path.write_text(json.dumps(serialized, indent=2))
