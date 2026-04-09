"""Tests for the bootstrap CI + ship-gate statistics helpers."""

from __future__ import annotations

import pytest

from evolution.meta_harness.statistics import (
    bootstrap_ci,
    mean_std,
    ship_verdict,
)


# ── mean_std ───────────────────────────────────────────────────────────


def test_mean_std_empty():
    assert mean_std([]) == (0.0, 0.0)


def test_mean_std_single():
    mean, std = mean_std([0.42])
    assert mean == 0.42
    assert std == 0.0


def test_mean_std_multiple():
    mean, std = mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    # Sample std (N-1): sqrt(10/4) = sqrt(2.5) ≈ 1.5811
    assert std == pytest.approx(1.5811, abs=0.001)


# ── bootstrap_ci ───────────────────────────────────────────────────────


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])


def test_bootstrap_ci_rejects_bad_confidence():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.5)


def test_bootstrap_ci_single_sample_is_degenerate():
    lo, hi = bootstrap_ci([0.42])
    assert lo == 0.42
    assert hi == 0.42


def test_bootstrap_ci_constant_samples_zero_width():
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5], seed=42)
    assert lo == 0.5
    assert hi == 0.5


def test_bootstrap_ci_brackets_the_mean():
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    lo, hi = bootstrap_ci(samples, seed=42)
    assert lo < 0.3 < hi  # true mean = 0.3
    assert lo >= 0.1
    assert hi <= 0.5


def test_bootstrap_ci_deterministic_with_seed():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci1 = bootstrap_ci(samples, seed=42)
    ci2 = bootstrap_ci(samples, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_wider_than_99_vs_90():
    samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lo_90, hi_90 = bootstrap_ci(samples, confidence=0.90, seed=42)
    lo_99, hi_99 = bootstrap_ci(samples, confidence=0.99, seed=42)
    # 99% interval should strictly contain the 90% interval
    assert lo_99 <= lo_90
    assert hi_99 >= hi_90


# ── ship_verdict ───────────────────────────────────────────────────────


def test_ship_verdict_empty_samples_is_no_ship():
    ship, stats = ship_verdict([])
    assert ship is False
    assert stats["n"] == 0
    assert "no samples" in stats["reason"]


def test_ship_verdict_all_positive_deltas_ships():
    """Treatment beats baseline on every single seed → clear SHIP."""
    deltas = [0.05, 0.08, 0.03, 0.06, 0.04]
    ship, stats = ship_verdict(deltas, seed=42)
    assert ship is True
    assert stats["ci_lower"] > 0
    assert stats["mean"] > 0


def test_ship_verdict_all_negative_deltas_does_not_ship():
    """Treatment LOSES on every seed → obvious NO-SHIP."""
    deltas = [-0.05, -0.08, -0.03, -0.06, -0.04]
    ship, stats = ship_verdict(deltas, seed=42)
    assert ship is False
    assert stats["ci_upper"] < 0


def test_ship_verdict_mixed_signs_noisy_no_ship():
    """Half wins, half losses — CI straddles zero → NO-SHIP."""
    deltas = [0.05, -0.04, 0.03, -0.06, 0.02, -0.01]
    ship, stats = ship_verdict(deltas, seed=42)
    # With only 6 noisy samples, 95% CI almost certainly crosses zero
    assert ship is False
    assert stats["ci_lower"] <= 0 <= stats["ci_upper"]


def test_ship_verdict_respects_min_ci_lower_threshold():
    """Even if CI lower > 0, a higher threshold can require better evidence."""
    deltas = [0.01, 0.02, 0.015, 0.018, 0.012]  # tiny positive improvements
    ship_loose, stats_loose = ship_verdict(deltas, min_ci_lower=0.0, seed=42)
    ship_strict, stats_strict = ship_verdict(deltas, min_ci_lower=0.05, seed=42)
    # Loose threshold ships; strict threshold doesn't
    assert ship_loose is True
    assert ship_strict is False


def test_ship_verdict_single_sample():
    """N=1 is degenerate but the function shouldn't crash."""
    ship, stats = ship_verdict([0.1])
    assert stats["n"] == 1
    assert stats["ci_lower"] == stats["ci_upper"] == 0.1


def test_ship_verdict_stats_fields_present():
    """Guarantee the stats dict has all expected fields for downstream reporting."""
    ship, stats = ship_verdict([0.1, 0.2, 0.3], seed=42)
    for field in ("mean", "std", "ci_lower", "ci_upper", "n", "confidence",
                  "min_ci_lower", "samples", "reason"):
        assert field in stats, f"missing field: {field}"
