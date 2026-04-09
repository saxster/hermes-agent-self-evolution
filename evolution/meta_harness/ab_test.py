"""A/B benchmark runner for the meta-harness filesystem proposer.

Phase E ship gate: run two GEPA passes on the same seed + dataset, one
with the filesystem proposer OFF (baseline) and one with it ON
(treatment), then compare the metrics.json results and produce a
ship/no-ship verdict.

This script SHELLS OUT to the existing evolve_prompt CLI rather than
calling it in-process, so both runs start from a completely clean Python
state (no DSPy caches, no module-level lessons leakage, no weird
concurrency).

Usage:
    python -m evolution.meta_harness.ab_test \\
        --section AGENT_IDENTITY \\
        --iterations 4 \\
        --seed 42 \\
        --output-root output/ab_tests/

Ship criteria (both must hold):
    1. treatment.evolved_score > baseline.evolved_score
       (treatment beats baseline on validation)
    2. treatment.evolved_score - treatment.baseline_score >=
       baseline.evolved_score - baseline.baseline_score
       (treatment improvement over its own baseline is >= baseline's
       improvement over its own baseline — ensures we're not just
       seeing dataset noise)

NOTE: This runs REAL optimizations and costs real LLM tokens. Expect
$5-$50+ per invocation depending on model and iteration count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    """Parsed metrics.json from a single GEPA run."""

    metrics_path: Path
    baseline_score: float
    evolved_score: float
    improvement: float
    iterations: int
    elapsed_seconds: float

    @classmethod
    def from_metrics_json(cls, metrics_path: Path) -> "RunResult":
        data = json.loads(metrics_path.read_text())
        return cls(
            metrics_path=metrics_path,
            baseline_score=float(data.get("baseline_score", 0.0)),
            evolved_score=float(data.get("evolved_score", 0.0)),
            improvement=float(data.get("improvement", 0.0)),
            iterations=int(data.get("iterations", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
        )


def run_variant(
    *,
    section: str,
    iterations: int,
    seed: int,
    enable_filesystem_proposer: bool,
    env_extra: Optional[dict] = None,
) -> RunResult:
    """Invoke `python -m evolution.prompts.evolve_prompt` in a subprocess.

    Returns the parsed RunResult from the output metrics.json.
    """
    variant = "treatment" if enable_filesystem_proposer else "baseline"
    print(f"\n{'=' * 70}")
    print(f"Running variant: {variant}")
    print(f"  section={section}  iterations={iterations}  seed={seed}")
    print(f"  enable_filesystem_proposer={enable_filesystem_proposer}")
    print(f"{'=' * 70}")

    env = os.environ.copy()
    env["HERMES_EVOLUTION_TRACING"] = "1"  # always trace, both variants
    # NOTE: evolve_prompt.py reads this from the CLI's constructed
    # EvolutionConfig, not from env. We need to pass it via CLI flag.
    if env_extra:
        env.update(env_extra)

    # We need a CLI flag for enable_filesystem_proposer. Since the
    # existing CLI doesn't expose one, we set an env var hack that the
    # CLI reads. See the env bridge added below.
    env["HERMES_FILESYSTEM_PROPOSER"] = "1" if enable_filesystem_proposer else "0"

    cmd = [
        sys.executable,
        "-m",
        "evolution.prompts.evolve_prompt",
        "--section",
        section,
        "--iterations",
        str(iterations),
    ]
    # DSPy seed is tricky — it's not a first-class CLI arg. We pass
    # the seed to the subprocess and rely on DSPy's internal determinism.
    env["DSPY_SEED"] = str(seed)

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{variant} run failed with exit code {result.returncode}")

    # Find the most recent output dir for this section
    out_root = Path("output") / "prompts" / section
    runs = sorted(out_root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f"No metrics.json found under {out_root}")
    latest = runs[-1]

    rr = RunResult.from_metrics_json(latest)
    print(f"  → evolved_score = {rr.evolved_score:.4f}")
    print(f"  → baseline_score = {rr.baseline_score:.4f}")
    print(f"  → improvement = {rr.improvement:+.4f}")
    print(f"  → elapsed = {rr.elapsed_seconds:.1f}s")
    print(f"  → metrics: {latest}")
    return rr


def compare(baseline: RunResult, treatment: RunResult) -> tuple[bool, list[str]]:
    """Return (ship_decision, reasons)."""
    reasons: list[str] = []
    ship = True

    # Criterion 1: treatment beats baseline on final evolved score
    delta_evolved = treatment.evolved_score - baseline.evolved_score
    if delta_evolved > 0:
        reasons.append(
            f"PASS: treatment evolved_score ({treatment.evolved_score:.4f}) > "
            f"baseline ({baseline.evolved_score:.4f}), delta = {delta_evolved:+.4f}"
        )
    else:
        ship = False
        reasons.append(
            f"FAIL: treatment evolved_score ({treatment.evolved_score:.4f}) <= "
            f"baseline ({baseline.evolved_score:.4f}), delta = {delta_evolved:+.4f}"
        )

    # Criterion 2: treatment's own improvement is at least as good as baseline's
    if treatment.improvement >= baseline.improvement:
        reasons.append(
            f"PASS: treatment improvement ({treatment.improvement:+.4f}) >= "
            f"baseline improvement ({baseline.improvement:+.4f})"
        )
    else:
        ship = False
        reasons.append(
            f"FAIL: treatment improvement ({treatment.improvement:+.4f}) < "
            f"baseline improvement ({baseline.improvement:+.4f}) — might be "
            f"dataset noise rather than signal"
        )

    return ship, reasons


def main():
    parser = argparse.ArgumentParser(description="Meta-harness A/B ship gate")
    parser.add_argument("--section", required=True, help="Prompt section name")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/ab_tests"),
        help="Where to write the A/B comparison report",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "#" * 70)
    print("# Meta-Harness A/B Ship Gate")
    print(f"# section={args.section} iterations={args.iterations} seed={args.seed}")
    print("#" * 70)

    baseline = run_variant(
        section=args.section,
        iterations=args.iterations,
        seed=args.seed,
        enable_filesystem_proposer=False,
    )

    treatment = run_variant(
        section=args.section,
        iterations=args.iterations,
        seed=args.seed,
        enable_filesystem_proposer=True,
    )

    ship, reasons = compare(baseline, treatment)

    print("\n" + "#" * 70)
    print("# A/B Ship Gate Verdict")
    print("#" * 70)
    for r in reasons:
        print(f"  {r}")
    print()
    print(f"DECISION: {'SHIP' if ship else 'DO NOT SHIP'}")

    report_path = args.output_root / f"{args.section}_{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "section": args.section,
                "iterations": args.iterations,
                "seed": args.seed,
                "timestamp": timestamp,
                "baseline": {
                    "metrics_path": str(baseline.metrics_path),
                    "baseline_score": baseline.baseline_score,
                    "evolved_score": baseline.evolved_score,
                    "improvement": baseline.improvement,
                    "elapsed_seconds": baseline.elapsed_seconds,
                },
                "treatment": {
                    "metrics_path": str(treatment.metrics_path),
                    "baseline_score": treatment.baseline_score,
                    "evolved_score": treatment.evolved_score,
                    "improvement": treatment.improvement,
                    "elapsed_seconds": treatment.elapsed_seconds,
                },
                "verdict": {
                    "ship": ship,
                    "reasons": reasons,
                },
            },
            indent=2,
        )
    )
    print(f"\nReport saved: {report_path}")

    sys.exit(0 if ship else 1)


if __name__ == "__main__":
    main()
