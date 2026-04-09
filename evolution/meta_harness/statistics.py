"""Statistical helpers for the meta-harness A/B ship gate.

Replaces single-run noisy point comparisons with multi-seed averaging
and bootstrap confidence intervals. Numpy-free — pure-stdlib so we
don't add a dependency to self-evolution just for A/B testing.

Design notes:
- Bootstrap over raw per-seed differences is the right statistic for a
  small N (3-10 seed pairs). Paired differences, not unpaired means,
  because each seed pair shares trainset/valset variance.
- We use percentile bootstrap (not BCa) because it's simple and N is
  small enough that the bias-correction barely moves the answer.
- Default confidence is 95%; default resamples is 10,000 which runs in
  ~10ms for N=10 on modern hardware.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def bootstrap_ci(
    samples: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the mean.

    Args:
        samples: Observed values (e.g. per-seed improvement deltas).
                 Must be non-empty.
        confidence: Two-sided confidence level in (0, 1). Default 0.95.
        n_resamples: Number of bootstrap resamples. Default 10,000.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (lower, upper) bound of the CI for the sample mean.

    Raises:
        ValueError: if samples is empty or confidence is out of range.
    """
    values = list(samples)
    if not values:
        raise ValueError("bootstrap_ci: samples must be non-empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"bootstrap_ci: confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    n = len(values)

    # Single-sample bootstrap is degenerate — return the value as a point
    if n == 1:
        return (values[0], values[0])

    resampled_means: list[float] = []
    for _ in range(n_resamples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(draw) / n)

    resampled_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    return (resampled_means[lo_idx], resampled_means[hi_idx])


def mean_std(samples: Sequence[float]) -> tuple[float, float]:
    """Return (mean, sample standard deviation) of a sequence.

    Uses N-1 denominator (Bessel's correction). Returns (0.0, 0.0) for
    empty input. Returns (mean, 0.0) for N=1.
    """
    values = list(samples)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return (mean, math.sqrt(variance))


def ship_verdict(
    paired_deltas: Sequence[float],
    confidence: float = 0.95,
    min_ci_lower: float = 0.0,
    seed: int | None = None,
) -> tuple[bool, dict]:
    """Compute a SHIP / NO-SHIP verdict from per-seed paired deltas.

    A "paired delta" is ``treatment_score - baseline_score`` for the SAME
    seed — i.e. the per-seed improvement attributable to the treatment.
    Pairing controls for seed-level variance that would otherwise dominate
    a small-N comparison.

    Ship criterion: the lower bound of the bootstrap CI for the mean
    paired delta must exceed ``min_ci_lower`` (default 0.0, meaning the
    treatment must be *probably better than zero improvement*).

    Returns:
        (ship, stats) where ``stats`` contains mean, std, ci_lower,
        ci_upper, n, and the raw samples list.
    """
    n = len(paired_deltas)
    if n == 0:
        return (False, {
            "mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "n": 0, "samples": [], "reason": "no samples",
        })

    mean, std = mean_std(paired_deltas)
    ci_lower, ci_upper = bootstrap_ci(
        paired_deltas, confidence=confidence, seed=seed,
    )
    ship = ci_lower > min_ci_lower

    return (ship, {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "confidence": confidence,
        "min_ci_lower": min_ci_lower,
        "samples": list(paired_deltas),
        "reason": (
            f"CI lower bound {ci_lower:+.4f} > {min_ci_lower}" if ship
            else f"CI lower bound {ci_lower:+.4f} <= {min_ci_lower}"
        ),
    })
