"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file body for optimization.

    **Design (Phase II of meta-harness roadmap, 2026-04):** the skill body
    text is bound as the signature's ``instructions`` field, NOT as an
    input field. DSPy optimizers (GEPA, MIPROv2, COPRO) mutate
    ``signature.instructions``; before this refactor the skill text was
    passed as an input and never mutated, so ``evolved_skill.txt`` was
    byte-identical to baseline. See ``PromptSectionModule`` for the full
    rationale. Frontmatter is handled separately via ``reassemble_skill``.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        sig = dspy.Signature(
            "task_input -> output",
            instructions=skill_text,
        )
        self.predictor = dspy.ChainOfThought(sig)

    @property
    def skill_text(self) -> str:
        """Read the current skill body text from the predictor's signature.

        After a GEPA run, this returns the EVOLVED skill body — not the
        baseline. ``reassemble_skill(frontmatter, module.skill_text)``
        produces the deployable full skill file.
        """
        try:
            return self.predictor.predict.signature.instructions or ""
        except Exception:  # pragma: no cover
            return ""

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
