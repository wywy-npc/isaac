"""Session persistence — JSONL transcript files."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import isaac.core.config as _cfg
from isaac.core.types import Message, Role, SessionState, ToolCall, ToolResult


def new_session(agent_name: str) -> SessionState:
    """Create a new session."""
    return SessionState(
        session_id=str(uuid.uuid4())[:8],
        agent_name=agent_name,
    )


def save_session(state: SessionState) -> Path:
    """Save session to JSONL transcript."""
    _cfg.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _cfg.SESSIONS_DIR / f"{state.agent_name}_{state.session_id}.jsonl"

    with open(path, "w") as f:
        # Header line
        f.write(json.dumps({
            "type": "session",
            "session_id": state.session_id,
            "agent_name": state.agent_name,
            "turn_count": state.turn_count,
            "total_tokens": state.total_tokens,
            "total_cost": state.total_cost,
            "summary": state.summary,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")

        # Message lines
        for msg in state.messages:
            line: dict[str, Any] = {
                "type": "message",
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.tool_calls:
                line["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "input": tc.input}
                    for tc in msg.tool_calls
                ]
            if msg.tool_results:
                line["tool_results"] = [
                    {"tool_call_id": tr.tool_call_id, "content": tr.content[:2000], "is_error": tr.is_error}
                    for tr in msg.tool_results
                ]
            if msg.meta:
                line["meta"] = msg.meta
            f.write(json.dumps(line) + "\n")

    return path


def load_session(agent_name: str, session_id: str) -> SessionState | None:
    """Load a session from JSONL transcript."""
    path = _cfg.SESSIONS_DIR / f"{agent_name}_{session_id}.jsonl"
    if not path.exists():
        return None

    state = SessionState(session_id=session_id, agent_name=agent_name)

    with open(path) as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if data.get("type") == "session":
                state.turn_count = data.get("turn_count", 0)
                state.total_tokens = data.get("total_tokens", 0)
                state.total_cost = data.get("total_cost", 0.0)
                state.summary = data.get("summary", "")
            elif data.get("type") == "message":
                msg = Message(
                    role=Role(data["role"]),
                    content=data.get("content", ""),
                )
                for tc_data in data.get("tool_calls", []):
                    msg.tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        input=tc_data["input"],
                    ))
                for tr_data in data.get("tool_results", []):
                    msg.tool_results.append(ToolResult(
                        tool_call_id=tr_data["tool_call_id"],
                        content=tr_data["content"],
                        is_error=tr_data.get("is_error", False),
                    ))
                msg.meta = data.get("meta", {})
                state.messages.append(msg)

    return state


def list_sessions(agent_name: str | None = None) -> list[dict[str, Any]]:
    """List available sessions."""
    _cfg.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions: list[dict[str, Any]] = []

    for path in sorted(_cfg.SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path) as f:
                header = json.loads(f.readline())
            if agent_name and header.get("agent_name") != agent_name:
                continue
            sessions.append({
                "session_id": header.get("session_id", ""),
                "agent_name": header.get("agent_name", ""),
                "turn_count": header.get("turn_count", 0),
                "total_cost": header.get("total_cost", 0.0),
                "timestamp": header.get("timestamp", ""),
            })
        except Exception:
            continue

    return sessions[:20]
