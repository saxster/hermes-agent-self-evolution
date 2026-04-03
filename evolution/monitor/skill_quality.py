"""Skill quality scoring from session history.

Queries the hermes-agent state.db (SQLite) for sessions where a skill
was active, then computes a rolling quality score based on proxy signals:
- User follow-up corrections (indicates the skill produced a bad answer)
- Tool call retries (indicates the tool call failed and had to be retried)
- Conversation length (unusually long conversations suggest confusion)
"""

import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Default state.db location
DEFAULT_STATE_DB = Path.home() / ".hermes" / "state.db"

# Heuristic thresholds
MAX_GOOD_TURN_COUNT = 8  # Conversations beyond this are "long"

# Market skills that produce predictions
_MARKET_PREDICTION_SKILLS = {
    "daily-market-briefing", "india-market-research",
}
CORRECTION_KEYWORDS = [
    "no, ", "not what i", "wrong", "incorrect", "that's not",
    "try again", "redo", "fix that", "actually,", "i meant",
]
RETRY_INDICATORS = [
    "tool_call_error", "retry", "failed to", "error:",
    "exception", "traceback",
]


@dataclass
class SkillScore:
    """Quality score for a single skill."""
    skill_name: str
    score: float  # 0.0 to 1.0
    session_count: int  # Number of sessions analyzed
    correction_rate: float  # Fraction of sessions with user corrections
    retry_rate: float  # Fraction of sessions with tool retries
    avg_turn_count: float  # Average conversation turns when skill was active
    window_days: int


class SkillQualityScorer:
    """Computes rolling quality scores for skills from session data.

    Reads from hermes-agent's state.db which tracks conversation sessions,
    messages, and tool calls.
    """

    def __init__(self, state_db_path: Optional[Path] = None):
        self.db_path = state_db_path or DEFAULT_STATE_DB

    def score(self, skill_name: str, window_days: int = 7) -> SkillScore:
        """Score a single skill based on recent session quality.

        Returns a SkillScore with a composite quality metric (0.0 - 1.0).
        """
        if not self.db_path.exists():
            return SkillScore(
                skill_name=skill_name,
                score=0.5,  # Neutral if no data
                session_count=0,
                correction_rate=0.0,
                retry_rate=0.0,
                avg_turn_count=0.0,
                window_days=window_days,
            )

        cutoff_ts = time.time() - (window_days * 86400)
        sessions = self._get_skill_sessions(skill_name, cutoff_ts)

        if not sessions:
            return SkillScore(
                skill_name=skill_name,
                score=0.5,
                session_count=0,
                correction_rate=0.0,
                retry_rate=0.0,
                avg_turn_count=0.0,
                window_days=window_days,
            )

        # Analyze each session
        correction_count = 0
        retry_count = 0
        total_turns = 0

        for session_id, messages in sessions.items():
            turn_count = len(messages)
            total_turns += turn_count

            has_correction = self._detect_corrections(messages)
            if has_correction:
                correction_count += 1

            has_retry = self._detect_retries(messages)
            if has_retry:
                retry_count += 1

        session_count = len(sessions)
        correction_rate = correction_count / session_count
        retry_rate = retry_count / session_count
        avg_turns = total_turns / session_count

        # Composite score: higher is better
        # Penalize corrections (weight 0.4), retries (0.3), long convos (0.3)
        correction_penalty = correction_rate * 0.4
        retry_penalty = retry_rate * 0.3
        length_penalty = min(1.0, max(0.0, (avg_turns - MAX_GOOD_TURN_COUNT) / MAX_GOOD_TURN_COUNT)) * 0.3

        total_penalty = correction_penalty + retry_penalty + length_penalty
        quality_score = max(0.0, min(1.0, 1.0 - total_penalty))

        # For market skills, incorporate prediction accuracy as additional signal
        if skill_name in _MARKET_PREDICTION_SKILLS:
            pred_accuracy = self._get_prediction_accuracy(cutoff_ts)
            if pred_accuracy is not None and pred_accuracy < 0.5:
                # Low prediction accuracy penalizes the skill score
                prediction_penalty = (0.5 - pred_accuracy) * 0.2
                quality_score = max(0.0, quality_score - prediction_penalty)

        return SkillScore(
            skill_name=skill_name,
            score=quality_score,
            session_count=session_count,
            correction_rate=correction_rate,
            retry_rate=retry_rate,
            avg_turn_count=avg_turns,
            window_days=window_days,
        )

    def get_all_scores(self, window_days: int = 7) -> dict[str, SkillScore]:
        """Score all skills that have been used in the given window.

        Returns a dict mapping skill_name -> SkillScore.
        """
        if not self.db_path.exists():
            return {}

        cutoff_ts = time.time() - (window_days * 86400)
        skill_names = self._get_active_skills(cutoff_ts)

        scores = {}
        for name in skill_names:
            scores[name] = self.score(name, window_days)
        return scores

    def _get_skill_sessions(
        self, skill_name: str, cutoff_ts: float,
    ) -> dict[str, list[dict]]:
        """Query state.db for sessions where skill was active.

        Returns {session_id: [messages]}.
        The schema is inferred from typical hermes-agent state.db tables.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find sessions that used this skill
            # The sessions table tracks which skills were attached
            cursor.execute("""
                SELECT DISTINCT s.id as session_id
                FROM sessions s
                WHERE s.skills LIKE ?
                  AND s.created_at > ?
            """, (f"%{skill_name}%", cutoff_ts))

            session_ids = [row["session_id"] for row in cursor.fetchall()]

            if not session_ids:
                conn.close()
                return {}

            # Get messages for each session
            sessions = {}
            placeholders = ",".join("?" * len(session_ids))
            cursor.execute(f"""
                SELECT session_id, role, content
                FROM messages
                WHERE session_id IN ({placeholders})
                ORDER BY session_id, created_at
            """, session_ids)

            for row in cursor.fetchall():
                sid = row["session_id"]
                if sid not in sessions:
                    sessions[sid] = []
                sessions[sid].append({
                    "role": row["role"],
                    "content": row["content"] or "",
                })

            conn.close()
            return sessions

        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            # Table schema might differ — return empty gracefully
            return {}

    def _get_active_skills(self, cutoff_ts: float) -> list[str]:
        """Get names of all skills used since cutoff timestamp."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT skills FROM sessions
                WHERE skills IS NOT NULL
                  AND skills != ''
                  AND created_at > ?
            """, (cutoff_ts,))

            skill_names = set()
            for row in cursor.fetchall():
                # skills column is comma-separated or JSON list
                skills_raw = row["skills"]
                if skills_raw.startswith("["):
                    import json
                    try:
                        names = json.loads(skills_raw)
                        skill_names.update(names)
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    for name in skills_raw.split(","):
                        stripped = name.strip()
                        if stripped:
                            skill_names.add(stripped)

            conn.close()
            return sorted(skill_names)

        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []

    def _detect_corrections(self, messages: list[dict]) -> bool:
        """Detect if the user corrected the agent in this session."""
        for msg in messages:
            if msg["role"] != "user":
                continue
            content_lower = msg["content"].lower()
            for keyword in CORRECTION_KEYWORDS:
                if keyword in content_lower:
                    return True
        return False

    def _detect_retries(self, messages: list[dict]) -> bool:
        """Detect if tool calls were retried in this session."""
        for msg in messages:
            content_lower = msg["content"].lower()
            for indicator in RETRY_INDICATORS:
                if indicator in content_lower:
                    return True
        return False

    def _get_prediction_accuracy(self, cutoff_ts: float) -> Optional[float]:
        """Get prediction accuracy for market skills from state.db predictions table.

        Returns accuracy as float (0.0-1.0), or None if insufficient data.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(correct) as correct_count "
                "FROM predictions WHERE resolved_at IS NOT NULL AND created_at >= ?",
                (cutoff_ts,),
            ).fetchone()
            conn.close()

            total = row["total"] if row else 0
            if total < 5:
                return None  # Not enough data

            correct_count = row["correct_count"] or 0
            return correct_count / total
        except Exception:
            return None  # predictions table may not exist
