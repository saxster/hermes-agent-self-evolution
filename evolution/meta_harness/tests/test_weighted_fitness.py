"""Tests for the weighted-fitness helpers in evolution.core.fitness.

These helpers are shared by skill_fitness_metric, prompt_section_fitness,
and tool_selection_fitness. Phase I of the "next level" roadmap makes
them actually USE the signal-informed weights from
signal_importers.get_signal_enhanced_fitness_weight().
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evolution.core.fitness import (
    DEFAULT_FITNESS_WEIGHTS,
    _conciseness_score,
    _correctness_score,
    _structure_score,
    compute_weighted_fitness,
    skill_fitness_metric,
)


# ── Sub-score components ───────────────────────────────────────────────


class TestCorrectnessScore:
    def test_empty_output_is_zero(self):
        assert _correctness_score("", "something") == 0.0
        assert _correctness_score("   ", "something") == 0.0

    def test_full_overlap_scores_high(self):
        out = "the quick brown fox jumps over lazy dog"
        exp = "the quick brown fox jumps over lazy dog"
        assert _correctness_score(out, exp) >= 0.95

    def test_zero_overlap_scores_low(self):
        out = "abc def ghi"
        exp = "xyz uvw rst"
        assert _correctness_score(out, exp) == pytest.approx(0.3, abs=0.01)

    def test_partial_overlap(self):
        out = "the quick brown fox"
        exp = "the quick brown fox jumps over lazy dog"
        # 4/8 overlap = 0.5 → score = 0.3 + 0.7*0.5 = 0.65
        assert _correctness_score(out, exp) == pytest.approx(0.65, abs=0.01)

    def test_empty_expected_neutral(self):
        assert _correctness_score("anything", "") == 0.5


class TestStructureScore:
    def test_empty_is_zero(self):
        assert _structure_score("") == 0.0
        assert _structure_score("   ") == 0.0

    def test_plain_text_baseline(self):
        assert _structure_score("just some plain prose") == pytest.approx(0.3, abs=0.01)

    def test_markdown_header_adds_points(self):
        score = _structure_score("## Header\nsome text")
        assert score > 0.3

    def test_fully_structured_response(self):
        text = (
            "## Summary\n\n"
            "Here are the key points:\n\n"
            "- **Point one**: something\n"
            "- **Point two**: another\n\n"
            "```python\ncode = True\n```\n"
        )
        assert _structure_score(text) >= 0.8


class TestConcisenessScore:
    def test_empty_is_zero(self):
        assert _conciseness_score("", "rubric") == 0.0

    def test_ideal_ratio_scores_high(self):
        exp = "expected rubric text approximately this long"
        out = exp  # 1.0× ratio
        assert _conciseness_score(out, exp) == 1.0

    def test_too_short_penalized(self):
        exp = "a long expected rubric with many detail requirements"
        out = "yes"
        # ratio ~0.06 → 0.2
        assert _conciseness_score(out, exp) == 0.2

    def test_moderate_verbosity_still_good(self):
        exp = "short"
        out = "a slightly more verbose response but not too bad"  # ratio ~10 → penalized
        score = _conciseness_score(out, exp)
        assert 0.0 <= score <= 1.0


# ── Weighted composite ─────────────────────────────────────────────────


class TestComputeWeightedFitness:
    def test_empty_output_is_zero(self):
        assert compute_weighted_fitness("", "rubric") == 0.0
        assert compute_weighted_fitness("   ", "rubric") == 0.0

    def test_default_weights_applied(self):
        out = "the quick brown fox jumps"
        exp = "the quick brown fox"
        score = compute_weighted_fitness(out, exp)
        # Should be a weighted combo of three sub-scores, not 0 or 1
        assert 0.0 < score < 1.0

    def test_correctness_heavy_weights_shift_score(self):
        """When correctness weight goes to 1.0, score equals correctness alone."""
        out = "the quick brown fox"
        exp = "the quick brown fox"
        weights = {"correctness_weight": 1.0, "procedure_weight": 0.0, "conciseness_weight": 0.0}
        score = compute_weighted_fitness(out, exp, weights=weights)
        assert score == pytest.approx(_correctness_score(out, exp), abs=0.01)

    def test_procedure_heavy_weights_reward_structure(self):
        """A structured output with zero keyword overlap should score
        higher when procedure_weight=1.0 than with defaults."""
        out = "## Analysis\n\n- point 1\n- point 2\n\n```code```"
        exp = "totally different keywords"
        default_score = compute_weighted_fitness(out, exp)
        procedure_score = compute_weighted_fitness(
            out,
            exp,
            weights={"correctness_weight": 0.0, "procedure_weight": 1.0, "conciseness_weight": 0.0},
        )
        assert procedure_score > default_score

    def test_missing_weights_keys_fall_back_to_default(self):
        """If weights dict is partial, missing keys use defaults."""
        out = "foo bar baz"
        exp = "foo bar"
        score_partial = compute_weighted_fitness(out, exp, weights={"correctness_weight": 0.9})
        # Should still produce a valid score (not crash on missing keys)
        assert 0.0 < score_partial <= 1.0

    def test_none_weights_uses_defaults(self):
        out = "some output"
        exp = "some rubric"
        score_none = compute_weighted_fitness(out, exp, weights=None)
        score_default = compute_weighted_fitness(out, exp, weights=DEFAULT_FITNESS_WEIGHTS)
        assert score_none == pytest.approx(score_default)


# ── Metric function (DSPy-compatible) ──────────────────────────────────


class TestSkillFitnessMetric:
    def test_basic_3_arg_call(self):
        """MIPROv2 calls with (example, prediction, trace)."""
        ex = SimpleNamespace(task_input="t", expected_behavior="expected output text")
        pred = SimpleNamespace(output="expected output text with extra")
        score = skill_fitness_metric(ex, pred, None)
        assert 0.0 < score <= 1.0

    def test_5_arg_gepa_call(self):
        """GEPA calls with (gold, pred, trace, pred_name, pred_trace)."""
        ex = SimpleNamespace(task_input="t", expected_behavior="expected output text")
        pred = SimpleNamespace(output="expected output text with extra")
        score = skill_fitness_metric(ex, pred, None, "pred_name_stub", {"some": "trace"})
        assert 0.0 < score <= 1.0

    def test_weights_kwarg_affects_score(self):
        """Passing weights must actually change the output."""
        ex = SimpleNamespace(task_input="t", expected_behavior="foo bar")
        pred = SimpleNamespace(output="## Heading\n- bullet\n\n```code```")

        # correctness-only: low because no keyword overlap
        correctness_only = skill_fitness_metric(
            ex, pred, None,
            weights={"correctness_weight": 1.0, "procedure_weight": 0.0, "conciseness_weight": 0.0},
        )
        # procedure-only: high because output is structured
        procedure_only = skill_fitness_metric(
            ex, pred, None,
            weights={"correctness_weight": 0.0, "procedure_weight": 1.0, "conciseness_weight": 0.0},
        )
        assert procedure_only > correctness_only

    def test_empty_output_returns_zero(self):
        ex = SimpleNamespace(task_input="t", expected_behavior="rubric")
        pred = SimpleNamespace(output="")
        assert skill_fitness_metric(ex, pred, None) == 0.0

    def test_missing_fields_safe(self):
        """Examples/predictions with missing fields shouldn't crash."""
        ex = SimpleNamespace()  # no task_input, no expected_behavior
        pred = SimpleNamespace()  # no output
        score = skill_fitness_metric(ex, pred, None)
        assert score == 0.0
