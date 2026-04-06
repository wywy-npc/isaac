"""Computer use — screenshot, click, type, scroll via pyautogui.

Handles the Anthropic computer_use tool responses. Each action returns
a screenshot for the model to see what happened.
"""
from __future__ import annotations

import asyncio
import base64
import io
from typing import Any


class ComputerController:
    """Thin wrapper around pyautogui for computer use tool."""

    def __init__(self, display_width: int = 1280, display_height: int = 800) -> None:
        self.width = display_width
        self.height = display_height

    async def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a computer use action. Returns tool result for the API."""
        loop = asyncio.get_event_loop()

        if action == "screenshot":
            return await loop.run_in_executor(None, self._screenshot)
        elif action == "click":
            x, y = kwargs.get("coordinate", (0, 0))
            return await loop.run_in_executor(None, self._click, x, y)
        elif action == "double_click":
            x, y = kwargs.get("coordinate", (0, 0))
            return await loop.run_in_executor(None, self._double_click, x, y)
        elif action == "type":
            text = kwargs.get("text", "")
            return await loop.run_in_executor(None, self._type, text)
        elif action == "key":
            key = kwargs.get("text", "")
            return await loop.run_in_executor(None, self._key, key)
        elif action == "scroll":
            x, y = kwargs.get("coordinate", (0, 0))
            delta_x = kwargs.get("delta_x", 0)
            delta_y = kwargs.get("delta_y", 0)
            return await loop.run_in_executor(None, self._scroll, x, y, delta_x, delta_y)
        elif action == "mouse_move":
            x, y = kwargs.get("coordinate", (0, 0))
            return await loop.run_in_executor(None, self._move, x, y)
        else:
            return {"error": f"Unknown action: {action}"}

    def _screenshot(self) -> dict[str, Any]:
        import pyautogui
        from PIL import Image
        img = pyautogui.screenshot()
        # Resize to match declared display size
        img = img.resize((self.width, self.height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"type": "image", "data": b64}

    def _click(self, x: int, y: int) -> dict[str, Any]:
        import pyautogui
        pyautogui.click(x, y)
        return self._screenshot()

    def _double_click(self, x: int, y: int) -> dict[str, Any]:
        import pyautogui
        pyautogui.doubleClick(x, y)
        return self._screenshot()

    def _type(self, text: str) -> dict[str, Any]:
        import pyautogui
        pyautogui.typewrite(text, interval=0.02)
        return self._screenshot()

    def _key(self, key: str) -> dict[str, Any]:
        import pyautogui
        pyautogui.hotkey(*key.split("+"))
        return self._screenshot()

    def _scroll(self, x: int, y: int, delta_x: int, delta_y: int) -> dict[str, Any]:
        import pyautogui
        pyautogui.moveTo(x, y)
        if delta_y:
            pyautogui.scroll(delta_y)
        return self._screenshot()

    def _move(self, x: int, y: int) -> dict[str, Any]:
        import pyautogui
        pyautogui.moveTo(x, y)
        return self._screenshot()
