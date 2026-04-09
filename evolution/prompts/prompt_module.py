"""Wraps a system prompt section as a DSPy module for optimization.

The system prompt in hermes-agent is assembled from named sections in
agent/prompt_builder.py. Each section is a Python string constant
(e.g. MEMORY_GUIDANCE, DEFAULT_AGENT_IDENTITY, CONTEXT_GRAPH_GUIDANCE).

This module reads those constants, wraps individual sections as DSPy
optimizable parameters, and patches evolved text back into the source.

Section layout (from run_agent.py _build_system_prompt):
  1. AGENT_IDENTITY  — DEFAULT_AGENT_IDENTITY or SOUL.md content
  2. MEMORY_GUIDANCE — guidance for the memory tool
  3. SESSION_SEARCH_GUIDANCE — guidance for session search
  4. SKILLS_GUIDANCE — guidance for skill creation/management
  5. CONTEXT_GRAPH_GUIDANCE — guidance for the context graph tool
  6. TOOL_USE_ENFORCEMENT_GUIDANCE — tells model to actually call tools
"""

import re
from pathlib import Path
from typing import Optional

import dspy


# Known section constants in prompt_builder.py
# Map from section name (CLI-friendly) to Python variable name
SECTION_CONSTANTS = {
    "AGENT_IDENTITY": "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE": "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE": "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE": "SKILLS_GUIDANCE",
    "CONTEXT_GRAPH_GUIDANCE": "CONTEXT_GRAPH_GUIDANCE",
    "TOOL_USE_ENFORCEMENT": "TOOL_USE_ENFORCEMENT_GUIDANCE",
}

# Sections that use SOUL.md as an override — AGENT_IDENTITY may come from
# a file rather than the constant. We handle both.
FILE_BACKED_SECTIONS = {
    "AGENT_IDENTITY": "SOUL.md",
}


def _prompt_builder_path(hermes_path: Path) -> Path:
    """Return the path to agent/prompt_builder.py."""
    return hermes_path / "agent" / "prompt_builder.py"


def _soul_md_path(hermes_path: Path) -> Optional[Path]:
    """Return the path to SOUL.md if it exists."""
    hermes_home = Path.home() / ".hermes"
    soul = hermes_home / "SOUL.md"
    if soul.exists():
        return soul
    return None


def load_prompt_sections(hermes_path: Path) -> dict:
    """Parse the system prompt into named sections.

    Returns:
        {
            "SECTION_NAME": {
                "var_name": str (Python variable name),
                "text": str (current section text),
                "source": "constant" | "file",
                "file_path": Path (source file),
            },
            ...
        }
    """
    pb_path = _prompt_builder_path(hermes_path)
    if not pb_path.exists():
        raise FileNotFoundError(f"prompt_builder.py not found at {pb_path}")

    source = pb_path.read_text()
    sections = {}

    for section_name, var_name in SECTION_CONSTANTS.items():
        text = _extract_string_constant(source, var_name)
        if text is not None:
            sections[section_name] = {
                "var_name": var_name,
                "text": text,
                "source": "constant",
                "file_path": pb_path,
            }

    # Check for SOUL.md override of AGENT_IDENTITY
    soul_path = _soul_md_path(hermes_path)
    if soul_path:
        soul_text = soul_path.read_text().strip()
        if soul_text:
            sections["AGENT_IDENTITY_SOUL"] = {
                "var_name": "SOUL.md",
                "text": soul_text,
                "source": "file",
                "file_path": soul_path,
            }

    return sections


def _extract_string_constant(source: str, var_name: str) -> Optional[str]:
    """Extract the value of a string constant from Python source.

    Handles:
      - VAR = "single line"
      - VAR = ("multi\\nline")
      - VAR = (\\n    "part1 "\\n    "part2"\\n)
    """
    # Pattern 1: VAR = ("..." ... ) — parenthesized string concatenation
    paren_pattern = re.compile(
        rf'^{re.escape(var_name)}\s*=\s*\(',
        re.MULTILINE,
    )
    match = paren_pattern.search(source)
    if match:
        start = match.end()
        # Find the matching closing paren
        depth = 1
        in_string = False
        string_char = None
        escape_next = False
        i = start

        while i < len(source) and depth > 0:
            ch = source[i]
            if escape_next:
                escape_next = False
                i += 1
                continue
            if ch == '\\' and in_string:
                escape_next = True
                i += 1
                continue
            if ch in ('"', "'") and not in_string:
                in_string = True
                string_char = ch
                i += 1
                continue
            if in_string and ch == string_char:
                in_string = False
                string_char = None
                i += 1
                continue
            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            i += 1

        if depth == 0:
            paren_content = source[start:i - 1]
            return _eval_concatenated_strings(paren_content)

    # Pattern 2: VAR = "single string"
    simple_pattern = re.compile(
        rf'^{re.escape(var_name)}\s*=\s*"((?:[^"\\]|\\.)*)"',
        re.MULTILINE,
    )
    match = simple_pattern.search(source)
    if match:
        return _unescape_python_string(match.group(1))

    # Pattern 3: VAR = 'single string'
    simple_sq_pattern = re.compile(
        rf"^{re.escape(var_name)}\s*=\s*'((?:[^'\\]|\\.)*)'",
        re.MULTILINE,
    )
    match = simple_sq_pattern.search(source)
    if match:
        return _unescape_python_string(match.group(1))

    return None


def _eval_concatenated_strings(paren_content: str) -> str:
    """Evaluate a parenthesized string expression like ("a " "b " "c").

    Handles implicit concatenation of adjacent string literals,
    which is the pattern used in prompt_builder.py.
    """
    # Extract all string literals
    parts = []
    pattern = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'')

    for match in pattern.finditer(paren_content):
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        parts.append(_unescape_python_string(raw))

    return "".join(parts)


def _unescape_python_string(s: str) -> str:
    """Unescape common Python string escape sequences."""
    result = s.replace('\\n', '\n')
    result = result.replace('\\t', '\t')
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace('\\\\', '\\')
    return result


class PromptSectionModule(dspy.Module):
    """A DSPy module that wraps one system prompt section for optimization.

    The section text is the parameter that GEPA optimizes.
    On each forward pass, the module uses the section as part of system
    instructions and evaluates agent response quality.
    """

    class TaskWithPromptSection(dspy.Signature):
        """Complete a task following the system prompt guidance.

        You are an AI agent with specific behavioral guidance injected
        into your system prompt. Follow the guidance naturally and
        complete the task.
        """
        system_guidance: str = dspy.InputField(desc="The system prompt section providing behavioral guidance")
        prior_lessons: str = dspy.InputField(desc="Failure patterns and candidate guidance distilled from prior GEPA iterations (may be empty on the first iteration). Use this to avoid repeating mistakes and to build on what has worked.")
        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the system guidance")

    def __init__(self, section_text: str):
        super().__init__()
        self.section_text = section_text
        self.predictor = dspy.ChainOfThought(self.TaskWithPromptSection)

    def forward(self, task_input: str) -> dspy.Prediction:
        # Meta-harness hook: if a trace writer is active, tell it which
        # candidate text produced this prediction. Cheap no-op otherwise.
        #
        # Candidate = (section_text, predictor_instructions). DSPy optimizers
        # (GEPA, MIPROv2) mutate the PREDICTOR's signature instructions, not
        # the module's section_text Python attribute, so we must hash both to
        # detect iteration boundaries correctly. See memory
        # reference_dspy_gepa_instrumentation.md for why.
        prior_lessons = ""
        try:
            from evolution.meta_harness.trace_writer import (
                get_active_writer,
                get_active_lessons,
            )
            _writer = get_active_writer()
            if _writer is not None:
                try:
                    _instr = self.predictor.predict.signature.instructions or ""
                except Exception:  # pragma: no cover
                    _instr = ""
                _composite = (
                    f"section_text:\n{self.section_text}\n\n"
                    f"predictor_instructions:\n{_instr}"
                )
                _writer.set_candidate(_composite)
            prior_lessons = get_active_lessons()
        except Exception:  # pragma: no cover — never block optimization
            prior_lessons = ""

        result = self.predictor(
            system_guidance=self.section_text,
            prior_lessons=prior_lessons,
            task_input=task_input,
        )
        return dspy.Prediction(output=result.output)


def reassemble_prompt(section_name: str, new_text: str, hermes_path: Path) -> bool:
    """Write an updated section back to its source file.

    For constant-backed sections: patches the Python string in prompt_builder.py.
    For file-backed sections (SOUL.md): writes the file directly.

    Returns True if successfully written, False otherwise.
    """
    sections = load_prompt_sections(hermes_path)

    if section_name not in sections:
        return False

    section = sections[section_name]

    # File-backed section — write directly
    if section["source"] == "file":
        section["file_path"].write_text(new_text + "\n")
        return True

    # Constant-backed — patch the Python source
    pb_path = section["file_path"]
    source = pb_path.read_text()
    old_text = section["text"]

    new_source = _replace_string_constant(source, section["var_name"], old_text, new_text)
    if new_source and new_source != source:
        pb_path.write_text(new_source)
        return True

    return False


def _replace_string_constant(
    source: str,
    var_name: str,
    old_text: str,
    new_text: str,
) -> Optional[str]:
    """Replace a string constant's value in Python source.

    Rebuilds the constant as a parenthesized multi-line string to
    maintain readability.
    """
    # Find the variable assignment
    paren_pattern = re.compile(
        rf'^({re.escape(var_name)}\s*=\s*)\(',
        re.MULTILINE,
    )
    match = paren_pattern.search(source)

    if match:
        prefix = match.group(1)
        # Find the end of the parenthesized expression
        start = match.start()
        paren_start = source.index('(', match.start() + len(prefix))
        depth = 1
        in_string = False
        string_char = None
        escape_next = False
        i = paren_start + 1

        while i < len(source) and depth > 0:
            ch = source[i]
            if escape_next:
                escape_next = False
                i += 1
                continue
            if ch == '\\' and in_string:
                escape_next = True
                i += 1
                continue
            if ch in ('"', "'") and not in_string:
                in_string = True
                string_char = ch
                i += 1
                continue
            if in_string and ch == string_char:
                in_string = False
                string_char = None
                i += 1
                continue
            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            i += 1

        if depth == 0:
            end = i
            # Build new parenthesized string
            new_assignment = _build_paren_string(prefix, new_text)
            return source[:start] + new_assignment + source[end:]

    # Try simple assignment: VAR = "..."
    simple_pattern = re.compile(
        rf'^({re.escape(var_name)}\s*=\s*)"(?:[^"\\]|\\.)*"',
        re.MULTILINE,
    )
    match = simple_pattern.search(source)
    if match:
        prefix = match.group(1)
        new_assignment = _build_paren_string(prefix, new_text)
        return source[:match.start()] + new_assignment + source[match.end():]

    return None


def _build_paren_string(prefix: str, text: str) -> str:
    """Build a parenthesized Python string constant.

    Splits on newlines to produce readable multi-line source like:
        VAR = (
            "line one\\n"
            "line two\\n"
        )
    """
    escaped = _escape_for_python(text)
    lines = escaped.split('\\n')

    if len(lines) <= 1:
        return f'{prefix}"{escaped}"'

    parts = []
    for i, line in enumerate(lines):
        suffix = "\\n" if i < len(lines) - 1 else ""
        parts.append(f'    "{line}{suffix}"')

    body = "\n".join(parts)
    return f"{prefix}(\n{body}\n)"


def _escape_for_python(text: str) -> str:
    """Escape text for embedding in a Python double-quoted string."""
    result = text.replace('\\', '\\\\')
    result = result.replace('"', '\\"')
    result = result.replace('\n', '\\n')
    result = result.replace('\t', '\\t')
    return result
