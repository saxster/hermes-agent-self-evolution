"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis
    # Signal-informed weights (defaults match original hardcoded values)
    _w_correctness: float = 0.5
    _w_procedure: float = 0.3
    _w_conciseness: float = 0.2

    @property
    def composite(self) -> float:
        """Weighted composite score.

        Weights can be overridden per-skill based on implicit signal data
        (e.g., skills with high correction rates get higher correctness weight).
        """
        raw = (
            self._w_correctness * self.correctness
            + self._w_procedure * self.procedure_following
            + self._w_conciseness * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


DEFAULT_FITNESS_WEIGHTS = {
    "correctness_weight": 0.5,
    "procedure_weight": 0.3,
    "conciseness_weight": 0.2,
}


def _structure_score(text: str) -> float:
    """Proxy for "follows expected procedure" — how structured is the output?

    Heuristic: presence of markdown structure (headings, lists, code blocks)
    indicates the agent followed a procedural format rather than freeform prose.
    Returns 0.0-1.0.
    """
    if not text or not text.strip():
        return 0.0
    signals = [
        ("##", 0.2),       # markdown headers
        ("- ", 0.15),       # bullet list
        ("1.", 0.1),        # numbered list
        ("```", 0.15),      # fenced code block
        ("**", 0.1),        # bold emphasis
        ("\n\n", 0.1),      # paragraph separation
    ]
    score = 0.3  # Base score for any non-empty text
    for signal, weight in signals:
        if signal in text:
            score += weight
    return min(1.0, score)


def _conciseness_score(output: str, expected: str) -> float:
    """Proxy for "appropriately concise" — penalize outputs drastically longer
    than the expected rubric suggests, but don't reward ultra-short answers.

    Returns 0.0-1.0. A ratio of 1.0-2.0× expected length scores highest;
    ratios >4× or <0.3× are penalized.
    """
    if not output:
        return 0.0
    if not expected:
        return 0.5  # No rubric length to compare against
    out_len = len(output)
    exp_len = max(1, len(expected))
    ratio = out_len / exp_len
    if ratio < 0.3:
        return 0.2  # Too short — likely missing required content
    if ratio <= 2.0:
        return 1.0  # Ideal range
    if ratio <= 4.0:
        return 0.8 - (ratio - 2.0) * 0.2  # Linear falloff from 0.8 → 0.4
    return 0.3  # Very verbose


def _correctness_score(output: str, expected: str) -> float:
    """Keyword overlap proxy — how many expected_behavior words appear in the output.

    Returns 0.0-1.0. This is the same heuristic the old skill_fitness_metric used.
    """
    if not output.strip():
        return 0.0
    if not expected.strip():
        return 0.5
    expected_words = set(expected.lower().split())
    output_words = set(output.lower().split())
    if not expected_words:
        return 0.5
    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, 0.3 + 0.7 * overlap)


def compute_weighted_fitness(
    agent_output: str,
    expected_behavior: str,
    weights: Optional[dict] = None,
) -> float:
    """Compute a weighted fitness score from agent output + rubric + weights.

    Shared by skill_fitness_metric, prompt_section_fitness, and
    tool_selection_fitness. Signal-informed weights (from
    signal_importers.get_signal_enhanced_fitness_weight) override the
    defaults to emphasize whichever dimension real user behavior showed
    was underperforming for that artifact.

    Returns a float in [0.0, 1.0].
    """
    if not agent_output or not agent_output.strip():
        return 0.0

    w = dict(DEFAULT_FITNESS_WEIGHTS)
    if weights:
        for k in ("correctness_weight", "procedure_weight", "conciseness_weight"):
            if k in weights:
                w[k] = float(weights[k])

    correctness = _correctness_score(agent_output, expected_behavior)
    procedure = _structure_score(agent_output)
    conciseness = _conciseness_score(agent_output, expected_behavior)

    composite = (
        w["correctness_weight"] * correctness
        + w["procedure_weight"] * procedure
        + w["conciseness_weight"] * conciseness
    )
    return max(0.0, min(1.0, composite))


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
    weights: Optional[dict] = None,
) -> float:
    """DSPy-compatible metric function for skill optimization.

    Accepts both the 3-arg MIPROv2 signature and the 5-arg GEPA signature
    (gold, pred, trace, pred_name, pred_trace) via default None values.

    Args:
        weights: Optional dict with correctness_weight, procedure_weight,
                 conciseness_weight from
                 signal_importers.get_signal_enhanced_fitness_weight().
                 When supplied, these override DEFAULT_FITNESS_WEIGHTS.

    Returns a float 0-1 score.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    return compute_weighted_fitness(agent_output, expected, weights=weights)


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
