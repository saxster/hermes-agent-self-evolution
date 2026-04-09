"""Evolve a Hermes Agent tool description using DSPy + GEPA.

Optimizes tool descriptions so that an LLM reliably selects the correct
tool for a given user task. The fitness metric measures tool selection
accuracy: given a task, does the optimized description lead to the model
picking this tool (and not a wrong one)?

Usage:
    python -m evolution.tools.evolve_tool_desc --tool web_search --iterations 10
    python -m evolution.tools.evolve_tool_desc --tool memory --iterations 5 --dry-run
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
from evolution.core.fitness import LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.tools.tool_module import (
    ToolDescModule,
    load_tool_desc,
    find_tool_file,
    reassemble_tool_desc,
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


# ── Fitness metric for tool description optimization ────────────────────────


def tool_selection_fitness(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """DSPy-compatible metric for tool description optimization.

    Scores whether the model's tool selection reasoning correctly identifies
    the target tool as the right choice (or correctly rejects it).

    The example should have:
      - task_input: the user's request
      - expected_behavior: which tool should be selected (tool name)

    The prediction should have:
      - output: the model's selection reasoning
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected_tool = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    output_lower = agent_output.lower()
    expected_lower = expected_tool.lower().strip()

    # Check if the expected tool name appears in the output
    tool_mentioned = expected_lower in output_lower

    # Check for positive selection signals
    positive_signals = [
        "select", "choose", "use", "correct tool", "right tool",
        "appropriate", "best fit", "should use", "recommend",
    ]
    has_positive = any(signal in output_lower for signal in positive_signals)

    # Check for negative/rejection signals
    negative_signals = [
        "not the right", "wrong tool", "incorrect", "should not",
        "wouldn't use", "not appropriate", "not suitable",
    ]
    has_negative = any(signal in output_lower for signal in negative_signals)

    if tool_mentioned and has_positive and not has_negative:
        return 1.0
    elif tool_mentioned and has_positive:
        return 0.7
    elif tool_mentioned:
        return 0.5
    elif has_positive:
        return 0.3
    else:
        return 0.1


# ── Synthetic dataset for tool selection ────────────────────────────────────


class ToolSelectionDatasetBuilder:
    """Generate (task, correct_tool) pairs from a tool description.

    Unlike skill datasets which test output quality, tool datasets test
    selection accuracy: "given this task, which tool should be used?"
    """

    class GenerateToolTasks(dspy.Signature):
        """Generate realistic user tasks that should trigger selection of a specific tool.

        Given a tool's name and description, generate diverse user requests where
        this tool is clearly the correct choice. Also generate some negative examples
        where a different tool would be more appropriate.

        Each test case should include:
        - task_input: a realistic user request
        - expected_behavior: the correct tool name to select
        - difficulty: easy, medium, hard
        - category: "positive" (this tool is correct) or "negative" (another tool is correct)
        """
        tool_name: str = dspy.InputField(desc="Name of the tool being tested")
        tool_description: str = dspy.InputField(desc="The tool's description text")
        all_tool_names: str = dspy.InputField(desc="Comma-separated list of all available tools")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateToolTasks)

    def generate(
        self,
        tool_name: str,
        tool_description: str,
        all_tool_names: list[str],
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a tool selection evaluation dataset."""
        import random
        from evolution.core.dataset_builder import EvalExample

        n = num_cases or self.config.eval_dataset_size
        tools_str = ", ".join(all_tool_names)

        lm = dspy.LM(self.config.judge_model)

        with dspy.context(lm=lm):
            result = self.generator(
                tool_name=tool_name,
                tool_description=tool_description,
                all_tool_names=tools_str,
                num_cases=n,
            )

        # Parse generated test cases
        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\[.*\]', result.test_cases, re.DOTALL)
            if match:
                cases_raw = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse test cases: {result.test_cases[:200]}")

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", tool_name),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "positive"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input")
        ]

        # Shuffle and split
        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


# ── Helper: discover all tool names ────────────────────────────────────────


def _discover_tool_names(hermes_path: Path) -> list[str]:
    """Scan hermes-agent/tools/*.py for all registered tool names."""
    import re

    tools_dir = hermes_path / "tools"
    if not tools_dir.exists():
        return []

    names = []
    register_pattern = re.compile(
        r"""registry\.register\(\s*\n?\s*name\s*=\s*['"](\w+)['"]""",
    )

    for py_file in tools_dir.glob("*.py"):
        try:
            content = py_file.read_text()
            for match in register_pattern.finditer(content):
                names.append(match.group(1))
        except Exception:
            continue

    return sorted(set(names))


# ── Main evolution function ─────────────────────────────────────────────────


def evolve(
    tool_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates tool description optimization."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the tool description ──────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving tool description: [bold]{tool_name}[/bold]\n")

    tool_info = load_tool_desc(tool_name, config.hermes_agent_path)
    if not tool_info:
        console.print(f"[red]✗ Tool '{tool_name}' not found in {config.hermes_agent_path / 'tools'}[/red]")
        sys.exit(1)

    original_desc = tool_info["description"]

    console.print(f"  File: {tool_info['file_path'].relative_to(config.hermes_agent_path)}")
    console.print(f"  Schema var: {tool_info['schema_var'] or '(inline)'}")
    console.print(f"  Description size: {len(original_desc)} chars")
    console.print(f"  Description: {original_desc[:120]}...")

    # Discover all tool names for context
    all_tool_names = _discover_tool_names(config.hermes_agent_path)
    console.print(f"  Available tools: {len(all_tool_names)} registered")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints (max {config.max_tool_desc_size} chars)")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "synthetic":
        builder = ToolSelectionDatasetBuilder(config)
        dataset = builder.generate(
            tool_name=tool_name,
            tool_description=original_desc,
            all_tool_names=all_tool_names,
        )
        save_path = Path("datasets") / "tools" / tool_name
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
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(original_desc, "tool_description")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline description has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create baseline module
    tools_str = ", ".join(all_tool_names)
    baseline_module = ToolDescModule(original_desc)

    # Prepare DSPy examples — add available_tools as input context
    trainset = _augment_examples(dataset.to_dspy_examples("train"), tools_str)
    valset = _augment_examples(dataset.to_dspy_examples("val"), tools_str)

    # ── Meta-harness tracing (opt-in via HERMES_EVOLUTION_TRACING=1) ────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "tools" / tool_name / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "traces"

    set_active_lessons("")  # always clear stale lessons

    trace_writer: Optional[TraceWriter] = None
    if tracing_enabled() or config.enable_filesystem_proposer:
        trace_writer = TraceWriter(
            archive_dir,
            artifact_name=tool_name,
        )
        set_active_writer(trace_writer)
        console.print(f"  [dim]Meta-harness tracing: ON → {archive_dir}[/dim]")

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
        make_tracing_metric(tool_selection_fitness, on_iteration_complete=diagnosis_cb)
        if trace_writer
        else tool_selection_fitness
    )

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        try:
            optimizer = dspy.GEPA(
                metric=metric_fn,
                max_steps=iterations,
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
                valset=valset,
            )
        except Exception as e:
            console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
            optimizer = dspy.MIPROv2(
                metric=metric_fn,
                auto="light",
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
            )
    finally:
        if trace_writer is not None:
            set_active_writer(None)
        set_active_lessons("")

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved description ──────────────────────────────────
    evolved_desc = optimized_module.description_text

    # ── 7. Validate evolved description ─────────────────────────────────
    console.print(f"\n[bold]Validating evolved description[/bold]")
    evolved_constraints = validator.validate_all(
        evolved_desc, "tool_description", baseline_text=original_desc,
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved description FAILED constraints — not deploying[/red]")
        output_path = Path("output") / "tools" / tool_name / "evolved_FAILED.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_desc)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = _augment_examples(dataset.to_dspy_examples("holdout"), tools_str)

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(
                task_input=ex.task_input,
                available_tools=tools_str,
            )
            baseline_score = tool_selection_fitness(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(
                task_input=ex.task_input,
                available_tools=tools_str,
            )
            evolved_score = tool_selection_fitness(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Tool Description Evolution Results")
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
    table.add_row(
        "Description Size",
        f"{len(original_desc)} chars",
        f"{len(evolved_desc)} chars",
        f"{len(evolved_desc) - len(original_desc):+} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    # output_dir was created before GEPA ran so the trace writer could
    # stream into it; reuse the same directory here.
    (output_dir / "evolved_description.txt").write_text(evolved_desc)
    (output_dir / "baseline_description.txt").write_text(original_desc)

    metrics = {
        "tool_name": tool_name,
        "timestamp": run_timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(original_desc),
        "evolved_size": len(evolved_desc),
        "max_size": config.max_tool_desc_size,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved tool description by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review: diff {output_dir}/baseline_description.txt {output_dir}/evolved_description.txt")
        console.print(f"  Apply:  python -c \"from evolution.tools.tool_module import reassemble_tool_desc; reassemble_tool_desc('{tool_name}', open('{output_dir}/evolved_description.txt').read(), Path('{config.hermes_agent_path}'))\"")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve description (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


def _augment_examples(examples: list[dspy.Example], available_tools: str) -> list[dspy.Example]:
    """Add available_tools field to DSPy examples for tool selection context."""
    augmented = []
    for ex in examples:
        new_ex = dspy.Example(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            available_tools=available_tools,
        ).with_inputs("task_input", "available_tools")
        augmented.append(new_ex)
    return augmented


# ── CLI ─────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--tool", required=True, help="Name of the tool to evolve (e.g. web_search, memory)")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(tool, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run):
    """Evolve a Hermes Agent tool description using DSPy + GEPA optimization."""
    evolve(
        tool_name=tool,
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
