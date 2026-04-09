"""Wraps a hermes-agent tool description as a DSPy module for optimization.

The key abstraction: a tool's description string becomes a parameterized DSPy
module where the description text is the optimizable parameter. GEPA can then
mutate the description and evaluate whether it improves tool selection accuracy.

Tool descriptions live in two places in hermes-agent:
  1. The `description` field in the schema dict (e.g. WEB_SEARCH_SCHEMA)
  2. The `description` kwarg passed to registry.register()

This module reads from the schema dict (source of truth) and patches it back.
"""

import ast
import re
from pathlib import Path
from typing import Optional

import dspy


def find_tool_file(tool_name: str, hermes_path: Path) -> Optional[Path]:
    """Find the Python file that registers a given tool.

    Searches tools/*.py for a registry.register() call with name=<tool_name>.
    """
    tools_dir = hermes_path / "tools"
    if not tools_dir.exists():
        return None

    target_pattern = re.compile(
        rf"""registry\.register\(\s*\n?\s*name\s*=\s*['"]({re.escape(tool_name)})['"]""",
        re.MULTILINE,
    )

    for py_file in tools_dir.glob("*.py"):
        try:
            content = py_file.read_text()
            if target_pattern.search(content):
                return py_file
        except Exception:
            continue

    return None


def load_tool_desc(tool_name: str, hermes_path: Path) -> Optional[dict]:
    """Load a tool's description from its Python source file.

    Looks for the description in two places (in priority order):
    1. The "description" key in the schema dict assigned to a variable
       ending in _SCHEMA (e.g. WEB_SEARCH_SCHEMA = {"description": "..."})
    2. The description= kwarg in the registry.register() call

    Returns:
        {
            "tool_name": str,
            "description": str,
            "file_path": Path,
            "schema_var": str or None (name of the schema variable),
            "raw_file": str (full file content),
        }
        or None if not found.
    """
    tool_file = find_tool_file(tool_name, hermes_path)
    if not tool_file:
        return None

    raw = tool_file.read_text()

    # Strategy 1: Find the schema dict variable that contains "description"
    # Look for TOOL_NAME_SCHEMA = { ... "description": "..." ... }
    schema_var = _find_schema_var_for_tool(raw, tool_name)
    if schema_var:
        desc = _extract_description_from_schema_var(raw, schema_var)
        if desc:
            return {
                "tool_name": tool_name,
                "description": desc,
                "file_path": tool_file,
                "schema_var": schema_var,
                "raw_file": raw,
            }

    # Strategy 2: Extract description= from registry.register() call
    desc = _extract_description_from_register(raw, tool_name)
    if desc:
        return {
            "tool_name": tool_name,
            "description": desc,
            "file_path": tool_file,
            "schema_var": None,
            "raw_file": raw,
        }

    return None


def _find_schema_var_for_tool(source: str, tool_name: str) -> Optional[str]:
    """Find the schema variable name referenced in registry.register(name=tool_name).

    Looks for patterns like:
        registry.register(name="web_search", ..., schema=WEB_SEARCH_SCHEMA, ...)
    and returns "WEB_SEARCH_SCHEMA".
    """
    # Find the register() call for this tool
    register_pattern = re.compile(
        rf"""registry\.register\([^)]*name\s*=\s*['"]({re.escape(tool_name)})['"][^)]*\)""",
        re.DOTALL,
    )
    match = register_pattern.search(source)
    if not match:
        return None

    call_text = match.group(0)

    # Extract schema= argument
    schema_match = re.search(r'schema\s*=\s*(\w+)', call_text)
    if schema_match:
        return schema_match.group(1)

    return None


def _extract_description_from_schema_var(source: str, var_name: str) -> Optional[str]:
    """Extract the 'description' value from a dict variable assignment.

    Handles both single-line and multi-line dict literals like:
        WEB_SEARCH_SCHEMA = {
            "name": "web_search",
            "description": "Search the web for information...",
            ...
        }
    """
    # Find the variable assignment — look for VAR_NAME = {
    var_pattern = re.compile(
        rf'^{re.escape(var_name)}\s*=\s*\{{',
        re.MULTILINE,
    )
    match = var_pattern.search(source)
    if not match:
        return None

    # Find the balanced closing brace
    start = match.start()
    brace_start = source.index('{', start)
    depth = 0
    in_string = False
    string_char = None
    escape_next = False

    for i in range(brace_start, len(source)):
        ch = source[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch in ('"', "'") and not in_string:
            in_string = True
            string_char = ch
            continue
        if ch == string_char and in_string:
            in_string = False
            string_char = None
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                dict_text = source[brace_start:i + 1]
                break
    else:
        return None

    # Extract "description" value from the dict text
    desc_pattern = re.compile(
        r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"|'
        r"""'description'\s*:\s*'((?:[^'\\]|\\.)*)'""",
    )
    desc_match = desc_pattern.search(dict_text)
    if desc_match:
        return desc_match.group(1) or desc_match.group(2)

    return None


def _extract_description_from_register(source: str, tool_name: str) -> Optional[str]:
    """Extract description= kwarg from registry.register() call."""
    register_pattern = re.compile(
        rf"""registry\.register\([^)]*name\s*=\s*['"]({re.escape(tool_name)})['"][^)]*\)""",
        re.DOTALL,
    )
    match = register_pattern.search(source)
    if not match:
        return None

    call_text = match.group(0)
    desc_match = re.search(
        r'description\s*=\s*"((?:[^"\\]|\\.)*)"|'
        r"""description\s*=\s*'((?:[^'\\]|\\.)*)'""",
        call_text,
    )
    if desc_match:
        return desc_match.group(1) or desc_match.group(2)

    return None


class ToolDescModule(dspy.Module):
    """A DSPy module that wraps a tool description for optimization.

    The tool description text is the parameter that GEPA optimizes.
    On each forward pass, the module simulates tool selection: given a
    user task, it uses the tool description to decide if this is the
    right tool, then produces a justification.
    """

    class ToolSelection(dspy.Signature):
        """Decide whether a tool should be selected for a given task.

        You are an AI agent deciding which tool to use. Read the tool description
        and determine if this tool is the right choice for the task. Explain your
        reasoning and state your selection decision.
        """
        tool_description: str = dspy.InputField(desc="The tool's description text")
        prior_lessons: str = dspy.InputField(desc="Failure patterns and description-writing guidance distilled from prior GEPA iterations (may be empty on the first iteration).")
        task_input: str = dspy.InputField(desc="The user's task or request")
        available_tools: str = dspy.InputField(desc="Comma-separated list of all available tool names")
        output: str = dspy.OutputField(desc="Your tool selection decision and reasoning")

    def __init__(self, description_text: str):
        super().__init__()
        self.description_text = description_text
        self.predictor = dspy.ChainOfThought(self.ToolSelection)

    def forward(self, task_input: str, available_tools: str = "") -> dspy.Prediction:
        # Meta-harness hook: if a trace writer is active, tell it which
        # candidate description produced this prediction. No-op otherwise.
        prior_lessons = ""
        try:
            from evolution.meta_harness.trace_writer import (
                get_active_writer,
                get_active_lessons,
            )
            _writer = get_active_writer()
            if _writer is not None:
                _writer.set_candidate(self.description_text)
            prior_lessons = get_active_lessons()
        except Exception:  # pragma: no cover — never block optimization
            prior_lessons = ""

        result = self.predictor(
            tool_description=self.description_text,
            prior_lessons=prior_lessons,
            task_input=task_input,
            available_tools=available_tools,
        )
        return dspy.Prediction(output=result.output)


def reassemble_tool_desc(tool_name: str, new_desc: str, hermes_path: Path) -> bool:
    """Patch a new description back into the tool's Python source file.

    Returns True if the file was successfully patched, False otherwise.

    Strategy:
    1. If a schema variable was identified, replace the description in the dict
    2. Otherwise, replace the description= kwarg in registry.register()
    """
    tool_info = load_tool_desc(tool_name, hermes_path)
    if not tool_info:
        return False

    source = tool_info["raw_file"]
    old_desc = tool_info["description"]

    # Escape special regex chars in old description
    escaped_old = re.escape(old_desc)

    # Replace in schema dict first (more precise)
    if tool_info["schema_var"]:
        pattern = re.compile(
            r'("description"\s*:\s*")' + escaped_old + r'(")',
        )
        new_source = pattern.sub(r'\g<1>' + _escape_for_replacement(new_desc) + r'\2', source, count=1)

        if new_source != source:
            tool_info["file_path"].write_text(new_source)
            return True

    # Fall back to replacing in register() call
    pattern = re.compile(
        r'(description\s*=\s*")' + escaped_old + r'(")',
    )
    new_source = pattern.sub(r'\g<1>' + _escape_for_replacement(new_desc) + r'\2', source, count=1)

    if new_source != source:
        tool_info["file_path"].write_text(new_source)
        return True

    return False


def _escape_for_replacement(text: str) -> str:
    """Escape a string for use in a regex replacement.

    Handles backslashes and double-quote escaping so the replacement
    is safe to insert into a Python string literal.
    """
    result = text.replace('\\', '\\\\')
    result = result.replace('"', '\\"')
    return result
