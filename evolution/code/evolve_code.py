"""Evolve a Hermes Agent tool file using LLM-proposed mutations.

Usage:
    python -m evolution.code.evolve_code --tool file_tools --iterations 5
    python -m evolution.code.evolve_code --tool web_tools --iterations 3 --dry-run
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
from evolution.core.constraints import ConstraintValidator
from evolution.code.code_organism import CodeOrganism, VariantResult

console = Console()


class CodeMutationProposer(dspy.Signature):
    """Propose a targeted improvement to a Python tool implementation.

    Analyze the code and propose ONE specific, testable mutation that improves:
    - Correctness (fix edge cases, handle errors better)
    - Performance (reduce unnecessary work, better algorithms)
    - Robustness (input validation, timeout handling, graceful degradation)
    - Clarity (better names, simpler logic, remove dead code)

    The mutation must:
    - Preserve ALL existing function signatures (no breaking changes)
    - Keep the same module-level exports
    - Be a focused, minimal change (not a rewrite)
    """
    original_code: str = dspy.InputField(desc="The current Python source code of the tool")
    tool_name: str = dspy.InputField(desc="Name of the tool being evolved")
    test_code: str = dspy.InputField(desc="The test file for this tool (empty if none)")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous mutation attempts (empty if first)")
    mutated_code: str = dspy.OutputField(desc="The complete mutated Python source code")
    mutation_description: str = dspy.OutputField(desc="One-line description of what changed and why")


def find_tool_file(tool_name: str, hermes_repo: Path) -> Optional[Path]:
    """Find a tool's Python file in the hermes-agent repo.

    Searches common tool locations:
    - hermes_agent/tools/<tool_name>.py
    - hermes_agent/tools/<tool_name>/<tool_name>.py
    - tools/<tool_name>.py
    """
    candidates = [
        hermes_repo / "hermes_agent" / "tools" / f"{tool_name}.py",
        hermes_repo / "hermes_agent" / "tools" / tool_name / f"{tool_name}.py",
        hermes_repo / "hermes_agent" / "tools" / tool_name / "__init__.py",
        hermes_repo / "tools" / f"{tool_name}.py",
        hermes_repo / "src" / "tools" / f"{tool_name}.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.relative_to(hermes_repo)

    # Broad search as fallback
    for py_file in hermes_repo.rglob(f"{tool_name}.py"):
        # Skip test files, __pycache__, and venv
        relative = py_file.relative_to(hermes_repo)
        path_str = str(relative)
        skip_dirs = ["__pycache__", ".venv", "venv", "node_modules", "test"]
        should_skip = any(d in path_str for d in skip_dirs)
        if not should_skip:
            return relative

    return None


def find_test_file(tool_name: str, hermes_repo: Path) -> Optional[Path]:
    """Find the test file for a tool."""
    candidates = [
        hermes_repo / "tests" / f"test_{tool_name}.py",
        hermes_repo / "tests" / "tools" / f"test_{tool_name}.py",
        hermes_repo / "tests" / "test_tools" / f"test_{tool_name}.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.relative_to(hermes_repo)

    return None


def evolve_code(
    tool_name: str,
    iterations: int = 5,
    optimizer_model: str = "openai/gpt-4.1",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    """Main code evolution function — proposes mutations and validates them."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        run_pytest=True,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    repo_path = config.hermes_agent_path

    # ── 1. Find tool and test files ────────────────────────────────────
    console.print(f"\n[bold cyan]Code Evolution[/bold cyan] — Evolving tool: [bold]{tool_name}[/bold]\n")

    tool_file = find_tool_file(tool_name, repo_path)
    if not tool_file:
        console.print(f"[red]Tool '{tool_name}' not found in {repo_path}[/red]")
        sys.exit(1)

    tool_path = repo_path / tool_file
    original_code = tool_path.read_text()
    console.print(f"  Tool file: {tool_file}")
    console.print(f"  Size: {len(original_code):,} chars, {original_code.count(chr(10))} lines")

    test_file = find_test_file(tool_name, repo_path)
    test_code = ""
    if test_file:
        test_code = (repo_path / test_file).read_text()
        console.print(f"  Test file: {test_file}")
    else:
        console.print("  [yellow]No test file found — mutations will only be AST-validated[/yellow]")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would propose {iterations} mutations via {optimizer_model}")
        console.print(f"  Would validate each via AST parse + pytest")
        return

    # ── 2. Set up mutation proposer ────────────────────────────────────
    console.print(f"\n[bold]Configuring LLM mutation proposer[/bold]")
    console.print(f"  Model: {optimizer_model}")
    console.print(f"  Iterations: {iterations}")

    lm = dspy.LM(optimizer_model)
    proposer = dspy.ChainOfThought(CodeMutationProposer)

    # ── 3. Set up code organism for branch management ──────────────────
    organism = CodeOrganism(config)

    # ── 4. Mutation loop ───────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running mutation loop ({iterations} iterations)...[/bold cyan]\n")

    best_variant: Optional[VariantResult] = None
    best_code: Optional[str] = None
    best_score: float = 0.0
    all_results: list[dict] = []
    feedback_history: str = ""

    start_time = time.time()

    for i in range(iterations):
        console.print(f"  [bold]Iteration {i + 1}/{iterations}[/bold]")

        # Propose a mutation
        with dspy.context(lm=lm):
            proposal = proposer(
                original_code=original_code,
                tool_name=tool_name,
                test_code=test_code,
                previous_feedback=feedback_history,
            )

        mutated_code = proposal.mutated_code
        description = proposal.mutation_description
        console.print(f"    Mutation: {description}")

        # Create variant branch
        variant = organism.create_variant(
            tool_file=tool_file,
            mutated_code=mutated_code,
            mutation_description=description,
            variant_id=i,
        )

        if not variant.ast_valid:
            console.print(f"    [red]AST parse failed — skipping[/red]")
            feedback_history += f"\nIteration {i + 1}: REJECTED — {variant.error}"
            all_results.append({
                "iteration": i + 1,
                "description": description,
                "ast_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
                "accepted": False,
            })
            continue

        # Validate on the branch
        validation = organism.validate_variant(
            variant.branch_name,
            test_file=test_file,
        )

        pass_rate = validation.pytest_pass_rate
        tests_ok = validation.tests_passed
        icon = "[green]PASS[/green]" if tests_ok else "[red]FAIL[/red]"
        console.print(f"    Tests: {icon} (pass rate: {pass_rate:.0%})")

        # Accept only if ALL tests pass (zero regression)
        accepted = tests_ok and pass_rate >= 1.0

        if accepted and pass_rate > best_score:
            best_variant = validation
            best_code = mutated_code
            best_score = pass_rate
            console.print(f"    [green]New best variant![/green]")

        # Build feedback for next iteration
        if accepted:
            feedback_history += f"\nIteration {i + 1}: ACCEPTED — {description}"
        else:
            failure_hint = validation.error or "Tests failed"
            # Include a snippet of test output for the LLM
            output_snippet = validation.test_output[-500:] if validation.test_output else ""
            feedback_history += (
                f"\nIteration {i + 1}: REJECTED — {description}. "
                f"Reason: {failure_hint}. Output: {output_snippet}"
            )

        all_results.append({
            "iteration": i + 1,
            "description": description,
            "ast_valid": True,
            "tests_passed": tests_ok,
            "pass_rate": pass_rate,
            "accepted": accepted,
        })

    elapsed = time.time() - start_time

    # ── 5. Clean up branches ───────────────────────────────────────────
    console.print(f"\n[bold]Cleaning up evolution branches[/bold]")
    deleted = organism.cleanup_branches()
    console.print(f"  Deleted {len(deleted)} branch(es)")

    # ── 6. Report results ──────────────────────────────────────────────
    table = Table(title="Code Evolution Results")
    table.add_column("Iter", style="bold", justify="right")
    table.add_column("Mutation")
    table.add_column("AST", justify="center")
    table.add_column("Tests", justify="center")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status", justify="center")

    for r in all_results:
        ast_icon = "[green]OK[/green]" if r["ast_valid"] else "[red]X[/red]"
        test_icon = "[green]OK[/green]" if r["tests_passed"] else "[red]X[/red]"
        status = "[green]ACCEPTED[/green]" if r["accepted"] else "[dim]rejected[/dim]"
        table.add_row(
            str(r["iteration"]),
            r["description"][:60],
            ast_icon,
            test_icon,
            f"{r['pass_rate']:.0%}",
            status,
        )

    console.print()
    console.print(table)

    # ── 7. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "code" / tool_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    (output_dir / "original.py").write_text(original_code)

    # Save best variant if we found one
    if best_code:
        (output_dir / "evolved.py").write_text(best_code)
        console.print(f"\n[bold green]Best variant saved to {output_dir}/evolved.py[/bold green]")
    else:
        console.print(f"\n[yellow]No valid mutations found — original code unchanged[/yellow]")

    # Save metrics
    metrics = {
        "tool_name": tool_name,
        "tool_file": str(tool_file),
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "original_size": len(original_code),
        "evolved_size": len(best_code) if best_code else len(original_code),
        "best_pass_rate": best_score,
        "elapsed_seconds": elapsed,
        "variants_tested": len(all_results),
        "variants_accepted": sum(1 for r in all_results if r["accepted"]),
        "results": all_results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"  Metrics saved to {output_dir}/metrics.json")
    console.print(f"  Total time: {elapsed:.1f}s")


@click.command()
@click.option("--tool", required=True, help="Name of the tool file to evolve (e.g. file_tools)")
@click.option("--iterations", default=5, help="Number of mutation attempts")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for proposing mutations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without running mutations")
def main(tool, iterations, optimizer_model, hermes_repo, dry_run):
    """Evolve a Hermes Agent tool file using LLM-proposed code mutations."""
    evolve_code(
        tool_name=tool,
        iterations=iterations,
        optimizer_model=optimizer_model,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
