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

        # Handle null scores gracefully — the constraint-failure path in
        # evolve_prompt.py writes `null` for baseline_score/evolved_score
        # when the evolved candidate fails constraints before holdout
        # evaluation runs. We treat those as 0.0 so the A/B runner can
        # still produce a delta (which will be 0.0 for that pair —
        # effectively "no signal from this seed").
        def _or_zero(val) -> float:
            try:
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        return cls(
            metrics_path=metrics_path,
            baseline_score=_or_zero(data.get("baseline_score")),
            evolved_score=_or_zero(data.get("evolved_score")),
            improvement=_or_zero(data.get("improvement")),
            iterations=int(data.get("iterations", 0) or 0),
            elapsed_seconds=_or_zero(data.get("elapsed_seconds")),
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


def run_multi_seed_ab(
    section: str,
    iterations: int,
    seeds: list[int],
) -> dict:
    """Run N baseline/treatment pairs, one per seed, and return aggregate stats.

    Returns a dict with:
      - per_seed: list of {seed, baseline, treatment, delta}
      - ship: bool (from bootstrap CI ship verdict)
      - stats: dict from ship_verdict (mean, std, ci_lower, ci_upper, ...)
    """
    from evolution.meta_harness.statistics import ship_verdict

    per_seed: list[dict] = []
    deltas: list[float] = []

    for i, seed in enumerate(seeds, start=1):
        print(f"\n{'#' * 70}")
        print(f"# Pair {i}/{len(seeds)} — seed={seed}")
        print(f"{'#' * 70}")

        baseline = run_variant(
            section=section,
            iterations=iterations,
            seed=seed,
            enable_filesystem_proposer=False,
        )
        treatment = run_variant(
            section=section,
            iterations=iterations,
            seed=seed,
            enable_filesystem_proposer=True,
        )

        delta = treatment.evolved_score - baseline.evolved_score
        per_seed.append({
            "seed": seed,
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
            "delta": delta,
        })
        deltas.append(delta)

        print(f"  pair {i} delta: {delta:+.4f}")

    ship, stats = ship_verdict(deltas)
    return {
        "per_seed": per_seed,
        "ship": ship,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Meta-harness A/B ship gate")
    parser.add_argument("--section", required=True, help="Prompt section name")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy single-seed mode (if --seeds is not given).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds for multi-seed mode, e.g. '42,43,44,45,46'. When given, overrides --seed and runs one pair per seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/ab_tests"),
        help="Where to write the A/B comparison report",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Multi-seed path (Phase III) ────────────────────────────────────
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        print("\n" + "#" * 70)
        print("# Meta-Harness A/B Ship Gate (MULTI-SEED)")
        print(f"# section={args.section} iterations={args.iterations} seeds={seeds}")
        print("#" * 70)

        result = run_multi_seed_ab(
            section=args.section,
            iterations=args.iterations,
            seeds=seeds,
        )

        stats = result["stats"]
        print("\n" + "#" * 70)
        print("# Multi-Seed A/B Ship Gate Verdict")
        print("#" * 70)
        print(f"  N pairs: {stats['n']}")
        print(f"  Per-seed deltas: {[f'{d:+.4f}' for d in stats['samples']]}")
        print(f"  Mean delta: {stats['mean']:+.4f}")
        print(f"  Std dev:    {stats['std']:.4f}")
        print(f"  {int(stats['confidence'] * 100)}% CI: [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}]")
        print(f"  Ship criterion: CI lower > {stats['min_ci_lower']}")
        print(f"  Verdict: {stats['reason']}")
        print()
        print(f"DECISION: {'SHIP' if result['ship'] else 'DO NOT SHIP'}")

        report_path = args.output_root / f"{args.section}_{timestamp}_multiseed.json"
        report_path.write_text(
            json.dumps({
                "section": args.section,
                "iterations": args.iterations,
                "seeds": seeds,
                "timestamp": timestamp,
                "per_seed": result["per_seed"],
                "stats": stats,
                "ship": result["ship"],
            }, indent=2)
        )
        print(f"\nReport saved: {report_path}")
        sys.exit(0 if result["ship"] else 1)

    # ── Legacy single-seed path ────────────────────────────────────────
    print("\n" + "#" * 70)
    print("# Meta-Harness A/B Ship Gate (SINGLE-SEED — noisy, consider --seeds for statistical rigor)")
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
