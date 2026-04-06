"""Sandbox backend registry — plugins register themselves here.

Instead of if/elif chains, backends register on import.
Builder looks them up by name: get_sandbox_backend("fly") -> FlySandbox.
"""
from __future__ import annotations

from typing import Any

_BACKENDS: dict[str, type] = {}


def register_sandbox(name: str, cls: type) -> None:
    """Register a sandbox backend by name."""
    _BACKENDS[name] = cls


def get_sandbox_backend(name: str) -> Any:
    """Get a sandbox backend instance by name. Auto-imports known backends."""
    if name not in _BACKENDS:
        # Auto-import known backends to trigger registration
        if name == "fly":
            from isaac.sandbox import fly  # noqa: F401 — triggers register_sandbox
        elif name == "e2b":
            from isaac.sandbox import e2b  # noqa: F401

    cls = _BACKENDS.get(name)
    if not cls:
        raise ValueError(f"Unknown sandbox backend: '{name}'. Available: {list(_BACKENDS.keys())}")
    return cls()


def list_backends() -> list[str]:
    """List registered backend names."""
    return list(_BACKENDS.keys())
