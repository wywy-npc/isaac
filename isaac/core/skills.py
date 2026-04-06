"""Skills — parameterized prompt recipes for multi-step workflows.

Skills are NOT code. They're markdown templates with {{params}} that get
rendered and injected into the conversation. The agent then follows the
instructions using its available tools.

Skills sit at the top of the 3-layer taxonomy:
  Skills (HOW — workflows)  →  Tools (WHAT — actions)  →  Connectors (WHERE — data + bundled tools)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """A reusable prompt recipe."""
    name: str
    description: str
    prompt_template: str                    # Markdown body with {{param}} placeholders
    params: dict[str, str] = field(default_factory=dict)  # param_name → description
    tools_used: list[str] = field(default_factory=list)   # Hints for which tools the skill needs
    user_invocable: bool = True             # Can user call via /name in REPL?


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    """Load all skills from a directory of markdown files with YAML frontmatter."""
    skills: dict[str, Skill] = {}

    if not skills_dir.exists():
        return skills

    for path in sorted(skills_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        try:
            skill = _parse_skill_file(path)
            if skill:
                skills[skill.name] = skill
        except Exception:
            continue

    return skills


def render_skill(skill: Skill, params: dict[str, str]) -> str:
    """Render a skill template by replacing {{param}} placeholders."""
    text = skill.prompt_template

    # Replace {{param}} with values, leave unreplaced ones as-is
    for key, value in params.items():
        text = text.replace(f"{{{{{key}}}}}", value)

    return text


def _parse_skill_file(path: Path) -> Skill | None:
    """Parse a skill markdown file with YAML frontmatter.

    Format:
    ---
    name: skill-name
    description: What this skill does
    params:
      topic: The subject to research
      depth: shallow, medium, deep
    tools_used: [web_search, memory_write]
    user_invocable: true
    ---

    # Skill prompt body with {{topic}} placeholders
    """
    content = path.read_text()

    # Split frontmatter from body
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None

    body = parts[2].strip()

    name = meta.get("name", path.stem)
    description = meta.get("description", "")
    params = meta.get("params", {})
    if isinstance(params, list):
        params = {p: "" for p in params}
    tools_used = meta.get("tools_used", [])
    user_invocable = meta.get("user_invocable", True)

    return Skill(
        name=name,
        description=description,
        prompt_template=body,
        params=params,
        tools_used=tools_used,
        user_invocable=user_invocable,
    )
