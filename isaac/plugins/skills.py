"""Skills plugin — registers use_skill and list_skills as agent tools.

The agent can discover skills via list_skills and activate them via use_skill.
When use_skill is called, the skill's prompt template is rendered and returned
as the tool result. The agent then follows those instructions using its tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from isaac.core.skills import Skill, load_skills, render_skill
from isaac.core.types import PermissionLevel, ToolDef


def build_skill_tools(skills_dir: Path | None = None) -> dict[str, tuple[ToolDef, Any]]:
    """Build skill tools: list_skills and use_skill."""
    from isaac.core.config import SKILLS_DIR
    _dir = skills_dir or SKILLS_DIR
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # Cache skills (reloaded on each list_skills call for hot-reload)
    _cache: dict[str, dict[str, Skill]] = {"skills": load_skills(_dir)}

    async def list_skills() -> dict[str, Any]:
        """List available skills with descriptions and parameters."""
        _cache["skills"] = load_skills(_dir)  # Hot-reload
        skills = _cache["skills"]
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "params": s.params,
                    "tools_used": s.tools_used,
                    "user_invocable": s.user_invocable,
                }
                for s in skills.values()
            ],
            "count": len(skills),
        }

    registry["list_skills"] = (
        ToolDef(
            name="list_skills",
            description="List available skills — reusable prompt workflows for complex tasks like research, repo onboarding, and bolt-on integration.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
            source="skill",
        ),
        list_skills,
    )

    async def use_skill(name: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """Activate a skill. Returns the rendered prompt — follow its instructions."""
        _cache["skills"] = load_skills(_dir)
        skills = _cache["skills"]

        skill = skills.get(name)
        if not skill:
            available = ", ".join(skills.keys()) if skills else "none"
            return {"error": f"Skill '{name}' not found. Available: {available}"}

        rendered = render_skill(skill, params or {})

        return {
            "skill": name,
            "instructions": rendered,
            "tools_needed": skill.tools_used,
            "note": "Follow the instructions above. Use the listed tools to complete the workflow.",
        }

    registry["use_skill"] = (
        ToolDef(
            name="use_skill",
            description=(
                "Activate a skill workflow. Returns detailed instructions to follow. "
                "Skills are multi-step prompt recipes that combine tools for complex tasks "
                "like deep research, repo bolt-on, or codebase onboarding."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name (from list_skills)"},
                    "params": {
                        "type": "object",
                        "description": "Skill parameters (varies per skill, see list_skills for details)",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
            source="skill",
        ),
        use_skill,
    )

    return registry
