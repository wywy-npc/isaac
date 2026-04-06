"""Personal memory store — facts about the user.

Reuses MemoryStore pointed at ~/.isaac/personal/. Same file format,
different directory, different conventions.

Recommended paths:
  preferences/         — user preferences (tools, workflows, models)
  context/             — background context (role, company, goals)
  people/              — people the user knows
  decisions/           — past decisions and reasoning
  facts/               — extracted facts from conversations
  logs/                — daily fact logs (auto-populated)
"""
from __future__ import annotations

from pathlib import Path

from isaac.core.config import PERSONAL_DIR
from isaac.memory.store import MemoryStore


def get_personal_store() -> MemoryStore:
    """Get a MemoryStore scoped to the personal memory directory."""
    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
    return MemoryStore(PERSONAL_DIR)
