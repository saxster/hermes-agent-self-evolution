"""Evolve a Hermes Agent system prompt section using DSPy + GEPA.

System prompt evolution is higher-risk than skill or tool description
evolution. A bad system prompt can degrade the entire agent. Conservative
constraints are enforced:
  - Max 10% growth over baseline (config.max_prompt_growth = 0.1)
  - Structural validation (no injection patterns, maintains key phrases)
  - Holdout evaluation must show improvement

Usage:
    python -m evolution.prompts.evolve_prompt --section MEMORY_GUIDANCE --iterations 5
    python -m evolution.prompts.evolve_prompt --section AGENT_IDENTITY --iterations 10 --dry-run
    python -m evolution.prompts.evolve_prompt --list-sections
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator, ConstraintResult
from evolution.prompts.prompt_module import (
    PromptSectionModule,
    load_prompt_sections,
    reassemble_prompt,
    SECTION_CONSTANTS,
)
from evolution.meta_harness.trace_writer import (
    TraceWriter,
    make_tracing_metric,
    set_active_writer,
    tracing_enabled,
    load_lessons_from_path,
    set_active_lessons,
)
from evolution.meta_harness.diagnose import DiagnosisAgent

console = Console()


# ── Prompt-specific fitness metric ──────────────────────────────────────────


def prompt_section_fitness(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """DSPy-compatible metric for prompt section optimization.

    Scores whether the agent's response (guided by the prompt section)
    exhibits the desired behavior. Uses keyword overlap as a fast proxy,
    same approach as skill_fitness_metric but with additional checks for
    behavioral compliance.

    Accepts the GEPA 5-arg signature ``(gold, pred, trace, pred_name,
    pred_trace)`` as well as the MIPROv2 3-arg signature via default
    ``None`` values. ``pred_name`` and ``pred_trace`` are ignored here
    — the fitness is computed only from ``example`` and ``prediction``.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    # Base score for non-empty output
    score = 0.4

    # Keyword overlap with expected behavior
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())

    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.5 * overlap)

    # Bonus for structured responses (system prompts often ask for structure)
    structure_signals = ["##", "- ", "1.", "2.", "```", "**"]
    has_structure = any(signal in agent_output for signal in structure_signals)
    if has_structure:
        score += 0.1

    # Penalty for refusals or off-topic responses
    refusal_signals = [
        "i cannot", "i'm unable", "as an ai", "i don't have",
        "not possible", "i apologize",
    ]
    has_refusal = any(signal in output_lower for signal in refusal_signals)
    if has_refusal:
        score -= 0.2

    return min(1.0, max(0.0, score))


# ── Prompt-specific constraint validation ───────────────────────────────────


def validate_prompt_section(
    section_text: str,
    section_name: str,
    baseline_text: str,
    config: EvolutionConfig,
) -> list[ConstraintResult]:
    """Validate an evolved prompt section with stricter constraints.

    In addition to standard size/growth checks, enforces:
    1. Conservative growth limit (10% instead of 20%)
    2. No prompt injection patterns
    3. Preserves key structural markers
    """
    results = []

    # Standard constraints via ConstraintValidator
    # Override growth limit to be more conservative for prompts
    conservative_config = EvolutionConfig(
        max_prompt_growth=0.1,  # 10% max growth
        max_skill_size=config.max_skill_size,
        max_tool_desc_size=config.max_tool_desc_size,
    )
    validator = ConstraintValidator(conservative_config)
    # NOTE: pass "prompt_section" (not "skill") so the YAML-frontmatter /
    # skill_structure check doesn't fire on prompt sections. Prompt sections
    # are plain strings with no name/description header.
    results.extend(validator.validate_all(
        section_text, "prompt_section", baseline_text=baseline_text,
    ))

    # Check for prompt injection patterns
    results.append(_check_no_injection(section_text))

    # Check that key phrases from baseline are preserved
    results.append(_check_key_phrases_preserved(section_text, baseline_text, section_name))

    return results


def _check_no_injection(text: str) -> ConstraintResult:
    """Ensure evolved prompt doesn't contain injection patterns."""
    import re

    injection_patterns = [
        r'ignore\s+(previous|all|above|prior)\s+instructions',
        r'system\s+prompt\s+override',
        r'disregard\s+(your|all|any)\s+(instructions|rules)',
        r'you\s+are\s+now\s+(?:a|an|the)\s+(?:different|new)',
        r'forget\s+(?:everything|all|your)',
        r'<\s*script\s*>',
        r'eval\s*\(',
        r'exec\s*\(',
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return ConstraintResult(
                passed=False,
                constraint_name="no_injection",
                message=f"Prompt contains injection pattern: {pattern}",
            )

    return ConstraintResult(
        passed=True,
        constraint_name="no_injection",
        message="No injection patterns detected",
    )


def _check_key_phrases_preserved(
    evolved: str,
    baseline: str,
    section_name: str,
) -> ConstraintResult:
    """Check that essential phrases from the baseline are preserved.

    Extracts "key phrases" — words that appear to be important identifiers
    or directives — and checks that most of them survive evolution.
    """
    # Section-specific key phrases that MUST be preserved
    required_phrases = {
        "AGENT_IDENTITY": ["hermes", "nous research"],
        "MEMORY_GUIDANCE": ["memory", "save"],
        "SESSION_SEARCH_GUIDANCE": ["session", "search", "conversation"],
        "SKILLS_GUIDANCE": ["skill"],
        "CONTEXT_GRAPH_GUIDANCE": ["context graph", "decision"],
        "TOOL_USE_ENFORCEMENT": ["tool"],
    }

    phrases = required_phrases.get(section_name, [])
    if not phrases:
        return ConstraintResult(
            passed=True,
            constraint_name="key_phrases",
            message="No required phrases for this section",
        )

    evolved_lower = evolved.lower()
    missing = [p for p in phrases if p not in evolved_lower]

    if missing:
        return ConstraintResult(
            passed=False,
            constraint_name="key_phrases",
            message=f"Missing required phrases: {', '.join(missing)}",
        )

    return ConstraintResult(
        passed=True,
        constraint_name="key_phrases",
        message=f"All {len(phrases)} required phrases preserved",
    )


# ── Main evolution function ─────────────────────────────────────────────────


def evolve(
    section_name: str,
    iterations: int = 5,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates prompt section optimization."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
        max_prompt_growth=0.1,  # Conservative: 10% max growth for prompts
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the prompt section ────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving prompt section: [bold]{section_name}[/bold]\n")

    sections = load_prompt_sections(config.hermes_agent_path)

    if section_name not in sections:
        console.print(f"[red]✗ Section '{section_name}' not found.[/red]")
        console.print(f"  Available sections: {', '.join(sorted(sections.keys()))}")
        sys.exit(1)

    section = sections[section_name]
    original_text = section["text"]

    console.print(f"  Source: {section['source']} ({section['file_path'].name})")
    console.print(f"  Variable: {section['var_name']}")
    console.print(f"  Size: {len(original_text):,} chars")
    console.print(f"  Preview: {original_text[:120]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Conservative constraints: max 10% growth, no injection, key phrases preserved")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=original_text,
            artifact_type="prompt_section",
        )
        save_path = Path("datasets") / "prompts" / section_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    baseline_constraints = validate_prompt_section(
        original_text, section_name, original_text, config,
    )
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")
    console.print(f"  [yellow]HIGH RISK[/yellow] — conservative constraints active (10% growth, structural validation)")

    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    baseline_module = PromptSectionModule(original_text)

    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── Meta-harness tracing (opt-in via HERMES_EVOLUTION_TRACING=1) ────
    # Create the output directory NOW (instead of post-run at step 10) so
    # the trace writer can stream per-task traces into it as GEPA runs.
    # This is the archive the Phase C diagnosis agent reads.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "prompts" / section_name / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "traces"

    # Always start with empty lessons — any stale value from a prior run
    # in the same process would poison this one.
    set_active_lessons("")

    trace_writer: Optional[TraceWriter] = None
    if tracing_enabled() or config.enable_filesystem_proposer:
        trace_writer = TraceWriter(
            archive_dir,
            artifact_name=section_name,
        )
        set_active_writer(trace_writer)
        console.print(f"  [dim]Meta-harness tracing: ON → {archive_dir}[/dim]")

    # Phase D callback: run the diagnosis agent between iterations and
    # install the lessons it writes as `prior_lessons` for the next pass.
    diagnosis_cb = None
    if config.enable_filesystem_proposer:
        def diagnosis_cb(iteration_num: int):  # noqa: E306
            if (iteration_num + 1) % max(1, config.diagnosis_interval) != 0:
                return
            if not archive_dir.exists():
                return
            console.print(
                f"  [dim]Meta-harness: running diagnosis after iteration {iteration_num}...[/dim]"
            )
            try:
                agent = DiagnosisAgent(
                    archive_dir=archive_dir,
                    model_name=config.diagnosis_model,
                    max_turns=config.diagnosis_max_turns,
                    max_cost_usd=config.max_diagnosis_budget_usd,
                )
                diag_result = agent.run()
                if diag_result.lessons_path is not None:
                    lessons_text = load_lessons_from_path(diag_result.lessons_path)
                    console.print(
                        f"  [dim]Diagnosis: wrote {len(lessons_text)} chars of lessons "
                        f"in {diag_result.turns_used} turns "
                        f"(${diag_result.total_cost_usd:.3f})[/dim]"
                    )
                else:
                    console.print(
                        f"  [yellow]Diagnosis: no lessons written "
                        f"({diag_result.stop_reason}, ${diag_result.total_cost_usd:.3f})[/yellow]"
                    )
            except Exception as diag_exc:  # noqa: BLE001
                console.print(f"  [yellow]Diagnosis failed: {diag_exc}[/yellow]")
        console.print(
            f"  [dim]Meta-harness filesystem proposer: ON "
            f"(diagnosis={config.diagnosis_model}, "
            f"budget=${config.max_diagnosis_budget_usd:.2f})[/dim]"
        )

    metric_fn = (
        make_tracing_metric(prompt_section_fitness, on_iteration_complete=diagnosis_cb)
        if trace_writer
        else prompt_section_fitness
    )

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        try:
            # GEPA signature: (metric, *, max_full_evals, reflection_lm, ...)
            # `iterations` CLI arg maps most naturally to max_full_evals — each full
            # eval runs the entire trainset once. `reflection_lm` is REQUIRED — GEPA
            # needs a strong model to reflect on candidates and propose new ones.
            # `optimizer_model` in EvolutionConfig is explicitly for this purpose.
            # The metric fitness functions accept GEPA's 5-arg signature via
            # pred_name/pred_trace defaulting to None.
            reflection_lm = dspy.LM(optimizer_model)
            optimizer = dspy.GEPA(
                metric=metric_fn,
                max_full_evals=iterations,
                reflection_lm=reflection_lm,
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
                valset=valset,
            )
            console.print(f"[dim]  Optimizer: dspy.GEPA (max_full_evals={iterations}, reflection_lm={optimizer_model})[/dim]")
        except Exception as e:
            # IMPORTANT: the fallback is an error-recovery path, not the happy path.
            # If you see this message, check the error — it's likely a version-skew
            # or signature change in dspy.GEPA, not "GEPA not available."
            console.print(
                f"[yellow]⚠ dspy.GEPA failed ({type(e).__name__}: {e}), "
                f"falling back to dspy.MIPROv2[/yellow]"
            )
            console.print(
                "[yellow]  Note: MIPROv2 tunes predictor instructions/few-shot,\n"
                "  not the section_text itself — meta-harness iteration counter\n"
                "  will not advance in this fallback path.[/yellow]"
            )
            optimizer = dspy.MIPROv2(
                metric=metric_fn,
                auto="light",
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
            )
    finally:
        # Always clear the active writer AND lessons so the next run
        # starts clean — the module-level lessons would otherwise leak.
        if trace_writer is not None:
            set_active_writer(None)
        set_active_lessons("")

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved section text ─────────────────────────────────
    evolved_text = optimized_module.section_text

    # ── 7. Validate evolved section (STRICT) ────────────────────────────
    console.print(f"\n[bold]Validating evolved section (strict mode)[/bold]")
    evolved_constraints = validate_prompt_section(
        evolved_text, section_name, original_text, config,
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved section FAILED constraints — not deploying[/red]")
        output_path = Path("output") / "prompts" / section_name / "evolved_FAILED.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_text)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = prompt_section_fitness(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score = prompt_section_fitness(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="System Prompt Section Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )

    growth_pct = (len(evolved_text) - len(original_text)) / max(1, len(original_text)) * 100
    table.add_row(
        "Section Size",
        f"{len(original_text):,} chars",
        f"{len(evolved_text):,} chars",
        f"{len(evolved_text) - len(original_text):+,} chars ({growth_pct:+.1f}%)",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    # output_dir was created before GEPA ran so the trace writer could
    # stream into it; reuse the same directory here.
    (output_dir / "evolved_section.txt").write_text(evolved_text)
    (output_dir / "baseline_section.txt").write_text(original_text)

    metrics = {
        "section_name": section_name,
        "var_name": section["var_name"],
        "source_type": section["source"],
        "timestamp": run_timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(original_text),
        "evolved_size": len(evolved_text),
        "growth_pct": growth_pct,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved section by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review: diff {output_dir}/baseline_section.txt {output_dir}/evolved_section.txt")
        console.print(f"  [yellow]⚠ Manual review strongly recommended before applying prompt changes[/yellow]")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve section (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


def list_sections(hermes_repo: Optional[str] = None):
    """Print all available prompt sections."""
    config = EvolutionConfig()
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print(f"\n[bold cyan]Available System Prompt Sections[/bold cyan]\n")

    try:
        sections = load_prompt_sections(config.hermes_agent_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    table = Table()
    table.add_column("Section", style="bold")
    table.add_column("Variable", style="dim")
    table.add_column("Source")
    table.add_column("Size", justify="right")
    table.add_column("Preview")

    for name, info in sorted(sections.items()):
        table.add_row(
            name,
            info["var_name"],
            info["source"],
            f"{len(info['text']):,} chars",
            info["text"][:60].replace("\n", " ") + "...",
        )

    console.print(table)


# ── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--section", default=None, help="Name of the prompt section to evolve (e.g. MEMORY_GUIDANCE)")
@click.option("--iterations", default=5, help="Number of GEPA iterations (default lower for safety)")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--list-sections", "list_only", is_flag=True, help="List available sections and exit")
def main(section, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run, list_only):
    """Evolve a Hermes Agent system prompt section using DSPy + GEPA optimization."""
    if list_only:
        list_sections(hermes_repo)
        return

    if not section:
        console.print("[red]✗ --section is required (or use --list-sections)[/red]")
        sys.exit(1)

    evolve(
        section_name=section,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
