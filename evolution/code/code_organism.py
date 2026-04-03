"""Manages code variants as git branches for safe mutation + validation.

A CodeOrganism wraps a single tool file and provides methods to:
1. Create variant branches with LLM-proposed mutations applied
2. Validate variants via AST parsing + pytest
3. Clean up evolution branches when done
"""

import ast
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from evolution.core.config import EvolutionConfig

console = Console()

BRANCH_PREFIX = "evolution/code/"


@dataclass
class VariantResult:
    """Result of creating and validating a code variant."""
    branch_name: str
    tool_file: str
    mutation_description: str
    ast_valid: bool = False
    tests_passed: bool = False
    test_output: str = ""
    pytest_pass_rate: float = 0.0
    error: Optional[str] = None


class CodeOrganism:
    """Manages code variants using git branches.

    Each mutation is applied on a dedicated branch so the main codebase
    stays untouched. Validation runs AST parse + pytest on the branch.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.repo_path = config.hermes_agent_path
        self.created_branches: list[str] = []

    def create_variant(
        self,
        tool_file: Path,
        mutated_code: str,
        mutation_description: str,
        variant_id: int = 0,
    ) -> VariantResult:
        """Create a git branch with the mutation applied to the tool file.

        Args:
            tool_file: Path to the tool file (relative to hermes-agent repo).
            mutated_code: The full mutated source code to write.
            mutation_description: Human-readable description of the mutation.
            variant_id: Numeric identifier for this variant.

        Returns:
            VariantResult with branch name and initial AST validation.
        """
        tool_name = tool_file.stem
        branch_name = f"{BRANCH_PREFIX}{tool_name}/v{variant_id}"

        result = VariantResult(
            branch_name=branch_name,
            tool_file=str(tool_file),
            mutation_description=mutation_description,
        )

        try:
            # Ensure we're on a clean state
            self._run_git(["stash", "--include-untracked"])

            # Create and switch to variant branch from current HEAD
            self._run_git(["checkout", "-b", branch_name])
            self.created_branches.append(branch_name)

            # Write the mutated code
            full_path = self.repo_path / tool_file
            full_path.write_text(mutated_code)

            # AST validation first (fast fail)
            ast_ok = self._validate_ast(mutated_code)
            result.ast_valid = ast_ok

            if not ast_ok:
                result.error = "AST parse failed — invalid Python syntax"
                self._run_git(["checkout", "-"])
                return result

            # Commit the change on the branch
            self._run_git(["add", str(tool_file)])
            self._run_git(["commit", "-m", f"evolution: {mutation_description}"])

        except Exception as e:
            result.error = f"Failed to create variant branch: {e}"
            # Try to get back to original branch
            try:
                self._run_git(["checkout", "-"])
            except Exception:
                pass

        finally:
            # Always return to original branch
            try:
                self._run_git(["checkout", "-"])
                self._run_git(["stash", "pop"])
            except Exception:
                pass

        return result

    def validate_variant(self, branch_name: str, test_file: Optional[Path] = None) -> VariantResult:
        """Run pytest on a variant branch and collect pass rate.

        Args:
            branch_name: The git branch to validate.
            test_file: Specific test file to run. If None, runs all tests.

        Returns:
            Updated VariantResult with test results.
        """
        result = VariantResult(
            branch_name=branch_name,
            tool_file="",
            mutation_description="",
        )

        original_branch = self._get_current_branch()

        try:
            self._run_git(["stash", "--include-untracked"])
            self._run_git(["checkout", branch_name])

            # Run pytest
            pytest_args = ["python", "-m", "pytest", "-q", "--tb=short"]
            if test_file:
                pytest_args.append(str(test_file))
            else:
                pytest_args.append("tests/")

            proc = subprocess.run(
                pytest_args,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.repo_path),
            )

            result.test_output = proc.stdout + proc.stderr
            result.tests_passed = proc.returncode == 0

            # Parse pass rate from pytest output
            result.pytest_pass_rate = self._parse_pass_rate(proc.stdout)
            result.ast_valid = True  # If we got this far, AST was fine

        except subprocess.TimeoutExpired:
            result.error = "pytest timed out (300s)"
            result.tests_passed = False
        except Exception as e:
            result.error = f"Validation failed: {e}"
            result.tests_passed = False
        finally:
            try:
                self._run_git(["checkout", original_branch])
                self._run_git(["stash", "pop"])
            except Exception:
                pass

        return result

    def cleanup_branches(self) -> list[str]:
        """Remove all evolution branches created during this run.

        Returns:
            List of branch names that were deleted.
        """
        deleted = []
        for branch in self.created_branches:
            try:
                self._run_git(["branch", "-D", branch])
                deleted.append(branch)
            except Exception as e:
                console.print(f"[yellow]Warning: could not delete branch {branch}: {e}[/yellow]")

        # Also clean up any leftover evolution branches not in our list
        try:
            proc = subprocess.run(
                ["git", "branch", "--list", f"{BRANCH_PREFIX}*"],
                capture_output=True,
                text=True,
                cwd=str(self.repo_path),
            )
            for line in proc.stdout.strip().split("\n"):
                branch = line.strip().lstrip("* ")
                if branch and branch not in deleted:
                    try:
                        self._run_git(["branch", "-D", branch])
                        deleted.append(branch)
                    except Exception:
                        pass
        except Exception:
            pass

        self.created_branches.clear()
        return deleted

    def _validate_ast(self, code: str) -> bool:
        """Check that the code parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a git command in the hermes-agent repo."""
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(self.repo_path),
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {error_msg}")
        return result

    def _get_current_branch(self) -> str:
        """Get the name of the current git branch."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def _parse_pass_rate(self, pytest_output: str) -> float:
        """Extract pass rate from pytest output.

        Parses lines like '5 passed, 2 failed' or '7 passed'.
        """
        import re

        passed = 0
        failed = 0
        errors = 0

        passed_match = re.search(r"(\d+) passed", pytest_output)
        if passed_match:
            passed = int(passed_match.group(1))

        failed_match = re.search(r"(\d+) failed", pytest_output)
        if failed_match:
            failed = int(failed_match.group(1))

        error_match = re.search(r"(\d+) error", pytest_output)
        if error_match:
            errors = int(error_match.group(1))

        total = passed + failed + errors
        if total == 0:
            return 0.0

        return passed / total
