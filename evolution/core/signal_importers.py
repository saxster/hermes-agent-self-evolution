"""Import implicit signal data from Hermes state.db for skill evolution.

Bridges the implicit_signals table (collected by agent/implicit_signal_collector.py)
to the evolution pipeline's EvalDataset format. This connects real user behavior
to skill optimization — the self-improving loop becomes truly automatic.

Usage from evolve_skill.py:
    python -m evolution.skills.evolve_skill --skill my-skill --eval-source signals

Standalone:
    python -m evolution.core.signal_importers --skill my-skill --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# hermes-agent root for imports
HERMES_AGENT_ROOT = Path(__file__).parent.parent.parent.parent / "hermes-agent"


def _get_hermes_db():
    """Get a SessionDB instance from the hermes-agent codebase."""
    sys.path.insert(0, str(HERMES_AGENT_ROOT))
    from hermes_state import SessionDB
    return SessionDB()


def get_skill_signals(
    skill_name: str,
    limit: int = 500,
) -> list[dict]:
    """Retrieve implicit signals for a specific skill.

    Returns list of signal dicts with signal_type, signal_value, signal_source,
    message_text, and metadata fields.
    """
    db = _get_hermes_db()
    try:
        # Get all skill_match signals for this skill
        all_signals = db.get_implicit_signals("skill_match", limit=limit)
        skill_signals = [
            s for s in all_signals
            if s.get("context_id") == skill_name
        ]
        return skill_signals
    finally:
        db.close()


def get_skill_performance_summary(days: int = 30) -> dict:
    """Get performance summary for all skills from implicit signals.

    Returns dict mapping skill_name to {total_suggestions, total_usages,
    total_skips, avg_signal_value, usage_rate, performance_score}.
    """
    sys.path.insert(0, str(HERMES_AGENT_ROOT))
    from hermes_state import SessionDB
    from agent.skill_performance import SkillPerformanceTracker

    db = SessionDB()
    try:
        tracker = SkillPerformanceTracker(db)
        report = tracker.generate_report(days=days)
        return {name: m.to_dict() for name, m in report.items()}
    finally:
        db.close()


def get_weakest_skills(top_n: int = 3, min_suggestions: int = 5, days: int = 30) -> list[dict]:
    """Get the weakest-performing skills ranked by performance score.

    Returns list of skill metric dicts, sorted worst-first.
    """
    sys.path.insert(0, str(HERMES_AGENT_ROOT))
    from hermes_state import SessionDB
    from agent.skill_performance import SkillPerformanceTracker

    db = SessionDB()
    try:
        tracker = SkillPerformanceTracker(db)
        weakest = tracker.get_weakest_skills(top_n=top_n, min_suggestions=min_suggestions, days=days)
        return [m.to_dict() for m in weakest]
    finally:
        db.close()


def get_signal_enhanced_fitness_weight(skill_name: str) -> dict:
    """Get signal-informed fitness weights for a skill.

    When real signals show that a skill is frequently skipped after matching,
    the fitness function should weigh "relevance" higher. When signals show
    corrections after skill use, weigh "correctness" higher.

    Returns dict with correctness_weight, procedure_weight, conciseness_weight.
    """
    signals = get_skill_signals(skill_name)

    if len(signals) < 5:
        # Not enough data — use defaults
        return {
            "correctness_weight": 0.5,
            "procedure_weight": 0.3,
            "conciseness_weight": 0.2,
        }

    total = len(signals)
    skip_count = sum(1 for s in signals if s.get("signal_source") == "implicit_skip")
    correction_count = sum(1 for s in signals if s.get("signal_source") == "implicit_correction")
    usage_count = sum(1 for s in signals if s.get("signal_source") == "implicit_usage")

    skip_rate = skip_count / total
    correction_rate = correction_count / max(correction_count + usage_count, 1)

    # If high skip rate → skill matches poorly → need better relevance/procedure
    if skip_rate > 0.5:
        return {
            "correctness_weight": 0.3,
            "procedure_weight": 0.5,
            "conciseness_weight": 0.2,
        }

    # If high correction rate → skill produces wrong answers → need better correctness
    if correction_rate > 0.3:
        return {
            "correctness_weight": 0.6,
            "procedure_weight": 0.25,
            "conciseness_weight": 0.15,
        }

    # Default balanced weights
    return {
        "correctness_weight": 0.5,
        "procedure_weight": 0.3,
        "conciseness_weight": 0.2,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check implicit signal data for skill evolution")
    parser.add_argument("--skill", help="Show signals for a specific skill")
    parser.add_argument("--weakest", type=int, default=0, help="Show N weakest skills")
    parser.add_argument("--summary", action="store_true", help="Show all-skills performance summary")
    parser.add_argument("--days", type=int, default=30, help="Lookback period (days)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.summary:
        summary = get_skill_performance_summary(days=args.days)
        if not summary:
            print("No skill performance data available yet.")
        else:
            print(f"Skill Performance Summary (last {args.days} days)")
            print("=" * 60)
            for name, metrics in sorted(summary.items(), key=lambda x: x[1]["performance_score"]):
                print(
                    f"  {name:30s}  score={metrics['performance_score']:.2f}  "
                    f"used={metrics['total_usages']}/{metrics['total_suggestions']}  "
                    f"skipped={metrics['total_skips']}"
                )

    elif args.weakest:
        weakest = get_weakest_skills(top_n=args.weakest, days=args.days)
        if not weakest:
            print("No skills with enough data for ranking.")
        else:
            print(f"Weakest {len(weakest)} skills (candidates for evolution):")
            for m in weakest:
                print(f"  {m['skill_name']:30s}  score={m['performance_score']:.2f}")

    elif args.skill:
        signals = get_skill_signals(args.skill)
        print(f"Signals for skill '{args.skill}': {len(signals)}")
        for s in signals[:10]:
            print(f"  type={s['signal_type']} value={s['signal_value']} source={s['signal_source']}")

        weights = get_signal_enhanced_fitness_weight(args.skill)
        print(f"Signal-informed fitness weights: {weights}")

    else:
        parser.print_help()
