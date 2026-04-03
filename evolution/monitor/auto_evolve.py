"""Consumes from EvolutionQueue and runs evolution on queued targets.

Picks the highest-priority pending item, runs the appropriate evolution
function (skill, tool, prompt), and posts results to mission-control
if configured.

Usage:
    python -m evolution.monitor.auto_evolve --once       # Process one item
    python -m evolution.monitor.auto_evolve --daemon      # Loop until queue empty
    python -m evolution.monitor.auto_evolve --daemon --interval 5m  # Loop with delay
"""

import json
import os
import time
import signal
import sys
from datetime import datetime
from typing import Optional

import click
from rich.console import Console

from evolution.monitor.evolution_queue import EvolutionQueue, QueueItem

console = Console()


def run_skill_evolution(item: QueueItem) -> dict:
    """Run skill evolution for a queued item.

    Imports and calls the Phase 1 evolve() function.
    Returns a results dict with success status and metrics.
    """
    from evolution.skills.evolve_skill import evolve

    try:
        evolve(
            skill_name=item.target_name,
            iterations=10,
            eval_source="synthetic",
            run_tests=False,
        )
        return {
            "success": True,
            "target": item.target_name,
            "type": "skill",
            "message": f"Skill evolution completed for {item.target_name}",
        }
    except SystemExit:
        # evolve() calls sys.exit on some errors
        return {
            "success": False,
            "target": item.target_name,
            "type": "skill",
            "message": f"Skill evolution failed for {item.target_name} (exit)",
        }
    except Exception as e:
        return {
            "success": False,
            "target": item.target_name,
            "type": "skill",
            "message": f"Skill evolution failed: {e}",
        }


def run_code_evolution(item: QueueItem) -> dict:
    """Run code evolution for a queued item.

    Imports and calls the Phase 4 evolve_code() function.
    """
    from evolution.code.evolve_code import evolve_code

    try:
        evolve_code(
            tool_name=item.target_name,
            iterations=5,
        )
        return {
            "success": True,
            "target": item.target_name,
            "type": "tool",
            "message": f"Code evolution completed for {item.target_name}",
        }
    except SystemExit:
        return {
            "success": False,
            "target": item.target_name,
            "type": "tool",
            "message": f"Code evolution failed for {item.target_name} (exit)",
        }
    except Exception as e:
        return {
            "success": False,
            "target": item.target_name,
            "type": "tool",
            "message": f"Code evolution failed: {e}",
        }


def post_to_mission_control(result: dict, item: QueueItem) -> bool:
    """Post evolution results to mission-control API.

    Only runs if MISSION_CONTROL_URL is set. Posts to
    POST /api/evolution/results with the result payload.

    Returns True if posted successfully, False otherwise.
    """
    mission_url = os.getenv("MISSION_CONTROL_URL")
    if not mission_url:
        return False

    import urllib.request
    import urllib.error

    endpoint = f"{mission_url.rstrip('/')}/api/evolution/results"

    payload = {
        "target_type": item.target_type,
        "target_name": item.target_name,
        "priority": item.priority,
        "reason": item.reason,
        "success": result["success"],
        "message": result["message"],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        request_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200 or resp.status == 201
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        console.print(f"  [yellow]Failed to post to mission-control: {e}[/yellow]")
        return False


def process_one(queue: EvolutionQueue) -> Optional[dict]:
    """Dequeue and process the highest-priority item.

    Returns the result dict, or None if queue was empty.
    """
    item = queue.dequeue()
    if not item:
        return None

    console.print(f"\n[bold]Processing:[/bold] {item.target_type}/{item.target_name}")
    console.print(f"  Priority: {item.priority}")
    console.print(f"  Reason: {item.reason}")
    console.print(f"  Age: {item.age_hours:.1f}h")

    # Dispatch based on target type
    dispatch = {
        "skill": run_skill_evolution,
        "tool": run_code_evolution,
    }

    handler = dispatch.get(item.target_type)
    if not handler:
        result = {
            "success": False,
            "target": item.target_name,
            "type": item.target_type,
            "message": f"Unknown target type: {item.target_type}",
        }
        queue.mark_complete(item.target_name, success=False, summary=result["message"])
        return result

    # Run evolution
    start_time = time.time()
    result = handler(item)
    elapsed = time.time() - start_time

    result["elapsed_seconds"] = elapsed

    # Mark complete in queue
    queue.mark_complete(
        item.target_name,
        success=result["success"],
        summary=result["message"],
    )

    # Report
    if result["success"]:
        console.print(f"  [green]Completed in {elapsed:.1f}s[/green]")
    else:
        console.print(f"  [red]Failed after {elapsed:.1f}s: {result['message']}[/red]")

    # Post to mission-control
    posted = post_to_mission_control(result, item)
    if posted:
        console.print("  [dim]Results posted to mission-control[/dim]")

    return result


def run_auto_evolve(
    once: bool = False,
    daemon: bool = False,
    interval_seconds: int = 60,
):
    """Main auto-evolution consumer loop."""
    queue = EvolutionQueue()

    # Graceful shutdown
    should_stop = False

    def handle_signal(signum, frame):
        nonlocal should_stop
        should_stop = True
        console.print("\n[yellow]Shutting down after current job[/yellow]")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    console.print("[bold cyan]Auto-Evolution Consumer[/bold cyan]")
    pending = queue.get_pending()
    console.print(f"  Queue: {len(pending)} pending item(s)")

    if once:
        # Process exactly one item
        result = process_one(queue)
        if result is None:
            console.print("[dim]Queue is empty — nothing to process[/dim]")
        return

    if daemon:
        console.print(f"  Mode: daemon (interval: {interval_seconds}s)")
        console.print()

        processed = 0
        while not should_stop:
            result = process_one(queue)
            if result is None:
                # Queue empty — wait and check again
                console.print("[dim]Queue empty — waiting for new items...[/dim]")
                sleep_end = time.time() + interval_seconds
                while time.time() < sleep_end and not should_stop:
                    remaining = sleep_end - time.time()
                    time.sleep(min(10, remaining))
                continue

            processed += 1

            # Brief pause between items to avoid hammering APIs
            if not should_stop:
                time.sleep(5)

        console.print(f"\n[bold]Daemon stopped. Processed {processed} item(s).[/bold]")
        return

    # Default: process all pending items and exit
    processed = 0
    while not should_stop:
        result = process_one(queue)
        if result is None:
            break
        processed += 1

    console.print(f"\n[bold]Processed {processed} item(s). Queue drained.[/bold]")


def parse_interval(interval_str: str) -> int:
    """Parse an interval string like '5m', '1h' into seconds."""
    interval_str = interval_str.strip().lower()
    if interval_str.endswith("h"):
        return int(float(interval_str[:-1]) * 3600)
    elif interval_str.endswith("m"):
        return int(float(interval_str[:-1]) * 60)
    elif interval_str.endswith("s"):
        return int(float(interval_str[:-1]))
    else:
        return int(float(interval_str))


@click.command()
@click.option("--once", is_flag=True, help="Process one item and exit")
@click.option("--daemon", is_flag=True, help="Run as daemon, continuously processing queue")
@click.option("--interval", default="5m", help="Poll interval in daemon mode (e.g. 5m, 1h)")
def main(once, daemon, interval):
    """Consume from the evolution queue and run evolution on targets."""
    interval_seconds = parse_interval(interval)
    run_auto_evolve(
        once=once,
        daemon=daemon,
        interval_seconds=interval_seconds,
    )


if __name__ == "__main__":
    main()
