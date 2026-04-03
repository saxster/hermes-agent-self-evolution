"""Continuous monitoring loop — watches skill quality and enqueues evolution.

Polls state.db on a configurable interval, computes rolling quality scores,
and enqueues evolution when a skill's score drops significantly from its
30-day baseline.

Usage:
    python -m evolution.monitor.monitor --interval 24h --dry-run
    python -m evolution.monitor.monitor --interval 6h
"""

import time
import signal
import sys
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.monitor.skill_quality import SkillQualityScorer, SkillScore
from evolution.monitor.evolution_queue import EvolutionQueue

console = Console()

# Threshold: if current score drops more than this fraction below baseline, trigger
DROP_THRESHOLD = 0.15  # 15% drop


def parse_interval(interval_str: str) -> int:
    """Parse an interval string like '24h', '6h', '30m' into seconds."""
    interval_str = interval_str.strip().lower()

    if interval_str.endswith("h"):
        return int(float(interval_str[:-1]) * 3600)
    elif interval_str.endswith("m"):
        return int(float(interval_str[:-1]) * 60)
    elif interval_str.endswith("s"):
        return int(float(interval_str[:-1]))
    elif interval_str.endswith("d"):
        return int(float(interval_str[:-1]) * 86400)
    else:
        # Assume seconds
        return int(float(interval_str))


def check_skill_quality(
    scorer: SkillQualityScorer,
    queue: EvolutionQueue,
    drop_threshold: float = DROP_THRESHOLD,
    dry_run: bool = False,
) -> list[dict]:
    """Run one quality check cycle across all active skills.

    Compares 7-day rolling scores against 30-day baselines.
    Enqueues evolution for skills that have dropped significantly.

    Returns list of findings (for reporting).
    """
    # Get current scores (7-day window)
    current_scores = scorer.get_all_scores(window_days=7)

    # Get baseline scores (30-day window)
    baseline_scores = scorer.get_all_scores(window_days=30)

    findings = []

    for skill_name, current in current_scores.items():
        baseline = baseline_scores.get(skill_name)

        if not baseline or baseline.session_count < 3:
            # Not enough data for a reliable baseline
            findings.append({
                "skill": skill_name,
                "current_score": current.score,
                "baseline_score": baseline.score if baseline else None,
                "drop": 0.0,
                "action": "skip_insufficient_data",
                "sessions": current.session_count,
            })
            continue

        if current.session_count < 2:
            # Not enough recent data
            findings.append({
                "skill": skill_name,
                "current_score": current.score,
                "baseline_score": baseline.score,
                "drop": 0.0,
                "action": "skip_few_recent_sessions",
                "sessions": current.session_count,
            })
            continue

        # Calculate drop as fraction of baseline
        drop = 0.0
        if baseline.score > 0:
            drop = (baseline.score - current.score) / baseline.score

        action = "ok"
        if drop > drop_threshold:
            action = "enqueue_evolution"

            # Compute priority based on drop severity
            # >30% drop = priority 1, >20% = priority 3, >15% = priority 5
            if drop > 0.30:
                priority = 1
            elif drop > 0.20:
                priority = 3
            else:
                priority = 5

            reason = (
                f"Quality dropped {drop:.0%} from baseline "
                f"({baseline.score:.2f} -> {current.score:.2f}, "
                f"{current.session_count} sessions in 7d)"
            )

            if not dry_run:
                enqueued = queue.enqueue(
                    target_type="skill",
                    target_name=skill_name,
                    priority=priority,
                    reason=reason,
                )
                if not enqueued:
                    action = "already_queued"

        findings.append({
            "skill": skill_name,
            "current_score": current.score,
            "baseline_score": baseline.score,
            "drop": drop,
            "action": action,
            "sessions": current.session_count,
        })

    return findings


def print_findings(findings: list[dict]):
    """Print a formatted table of monitoring findings."""
    if not findings:
        console.print("  [dim]No active skills found[/dim]")
        return

    table = Table(title="Skill Quality Report")
    table.add_column("Skill", style="bold")
    table.add_column("Current", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Drop", justify="right")
    table.add_column("Sessions", justify="right")
    table.add_column("Action")

    for f in sorted(findings, key=lambda x: x.get("drop", 0), reverse=True):
        current_str = f"{f['current_score']:.2f}"
        baseline_str = f"{f['baseline_score']:.2f}" if f["baseline_score"] is not None else "-"

        drop_val = f["drop"]
        if drop_val > DROP_THRESHOLD:
            drop_str = f"[red]{drop_val:.0%}[/red]"
        elif drop_val > 0.05:
            drop_str = f"[yellow]{drop_val:.0%}[/yellow]"
        else:
            drop_str = f"[green]{drop_val:.0%}[/green]"

        action = f["action"]
        action_style_map = {
            "enqueue_evolution": "[red bold]ENQUEUE[/red bold]",
            "already_queued": "[yellow]already queued[/yellow]",
            "ok": "[green]OK[/green]",
            "skip_insufficient_data": "[dim]insufficient data[/dim]",
            "skip_few_recent_sessions": "[dim]few recent sessions[/dim]",
        }
        action_str = action_style_map.get(action, action)

        table.add_row(
            f["skill"],
            current_str,
            baseline_str,
            drop_str,
            str(f["sessions"]),
            action_str,
        )

    console.print(table)


def run_monitor(
    interval_seconds: int,
    dry_run: bool = False,
    once: bool = False,
    drop_threshold: float = DROP_THRESHOLD,
):
    """Main monitoring loop."""
    scorer = SkillQualityScorer()
    queue = EvolutionQueue()

    # Graceful shutdown
    should_stop = False

    def handle_signal(signum, frame):
        nonlocal should_stop
        should_stop = True
        console.print("\n[yellow]Received shutdown signal — finishing current cycle[/yellow]")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    console.print(f"[bold cyan]Evolution Monitor[/bold cyan]")
    if dry_run:
        console.print("[yellow]DRY RUN — will not enqueue anything[/yellow]")
    interval_display = f"{interval_seconds // 3600}h" if interval_seconds >= 3600 else f"{interval_seconds // 60}m"
    console.print(f"  Interval: {interval_display}")
    console.print(f"  Drop threshold: {drop_threshold:.0%}")
    console.print(f"  Queue: {queue.queue_path}")
    console.print()

    cycle = 0
    while not should_stop:
        cycle += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(f"[bold]Cycle {cycle}[/bold] — {timestamp}")

        findings = check_skill_quality(
            scorer=scorer,
            queue=queue,
            drop_threshold=drop_threshold,
            dry_run=dry_run,
        )
        print_findings(findings)

        enqueued_count = sum(1 for f in findings if f["action"] == "enqueue_evolution")
        if enqueued_count > 0:
            console.print(f"\n  [red]Enqueued {enqueued_count} skill(s) for evolution[/red]")

        # Prune old completed items
        pruned = queue.prune_completed()
        if pruned > 0:
            console.print(f"  Pruned {pruned} old queue items")

        if once:
            break

        console.print(f"\n  Next check in {interval_display}\n")

        # Sleep in short increments so we can respond to signals
        sleep_end = time.time() + interval_seconds
        while time.time() < sleep_end and not should_stop:
            remaining = sleep_end - time.time()
            time.sleep(min(10, remaining))

    console.print("[bold]Monitor stopped.[/bold]")


@click.command()
@click.option("--interval", default="24h", help="Check interval (e.g. 24h, 6h, 30m)")
@click.option("--dry-run", is_flag=True, help="Report findings without enqueuing")
@click.option("--once", is_flag=True, help="Run one check cycle and exit")
@click.option("--drop-threshold", default=0.15, help="Quality drop fraction to trigger evolution (default 0.15)")
def main(interval, dry_run, once, drop_threshold):
    """Monitor skill quality and enqueue evolution for degraded skills."""
    interval_seconds = parse_interval(interval)
    run_monitor(
        interval_seconds=interval_seconds,
        dry_run=dry_run,
        once=once,
        drop_threshold=drop_threshold,
    )


if __name__ == "__main__":
    main()
