"""Toolsmith — auto-generate tools from natural language descriptions.

Uses Claude to write tool files, validates them, and drops them
in ~/.isaac/tools/ for hot-reload by the unified MCP server.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import anthropic

from isaac.core.config import TOOLS_DIR, get_env


TOOLSMITH_PROMPT = """You are a toolsmith. Write an ISAAC tool file.

The tool file must follow this exact format:

```python
TOOLS = [
    {{
        "name": "tool_name",
        "description": "What it does",
        "params": {{"arg1": str, "arg2": int}},
        "handler": None,  # set below
    }}
]

async def tool_handler(arg1: str, arg2: int = 0) -> dict:
    # implementation
    return {{"result": "..."}}

TOOLS[0]["handler"] = tool_handler
```

Rules:
- TOOLS must be a list at module level
- Each tool needs name, description, params dict, and handler
- Handlers must be async functions returning a dict
- Use httpx for HTTP requests (already available)
- Keep it minimal — one file, working out of the box
- No external dependencies beyond httpx and the standard library

Now write a tool file for: {description}

Output ONLY the Python code, no explanation."""


class Toolsmith:
    """Generates tool files from natural language descriptions."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=get_env("ANTHROPIC_API_KEY"))

    async def generate(self, description: str) -> dict[str, Any]:
        """Generate a plugin from a description. Returns {code, tool_name, errors}."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": TOOLSMITH_PROMPT.format(description=description),
            }],
        )

        raw = response.content[0].text if response.content else ""
        code = self._extract_code(raw)
        errors = self._validate(code)
        tool_name = self._extract_tool_name(code)

        return {
            "code": code,
            "tool_name": tool_name,
            "errors": errors,
        }

    def save(self, result: dict[str, Any]) -> Path:
        """Save generated tool to tools directory."""
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        name = result.get("tool_name", "generated_tool")
        path = TOOLS_DIR / f"{name}.py"
        path.write_text(result["code"])
        return path

    @staticmethod
    def _extract_code(text: str) -> str:
        """Strip markdown code fences if present."""
        # Match ```python ... ``` or ``` ... ```
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _validate(code: str) -> list[str]:
        """Validate plugin code. Returns list of errors (empty = valid)."""
        errors: list[str] = []

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]

        # Check for TOOLS list
        has_tools = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TOOLS":
                        has_tools = True
        if not has_tools:
            errors.append("Missing TOOLS list at module level")

        # Check for at least one async function
        has_async = any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(tree))
        if not has_async:
            errors.append("No async handler function found")

        return errors

    @staticmethod
    def _extract_tool_name(code: str) -> str | None:
        """Extract the first tool name from the TOOLS list."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', code)
        return match.group(1) if match else None
