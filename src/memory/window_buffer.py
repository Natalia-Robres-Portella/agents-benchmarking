"""Sliding-window memory — keeps the last N observations in a deque."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from src.memory.base import MemoryModule


class WindowBufferMemory(MemoryModule):
    """
    Stores the most recent `window_size` memory entries.
    read() ignores the query and returns the newest entries first.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window: Deque[str] = deque(maxlen=window_size)

    def read(self, query: str, k: int = 5) -> List[str]:
        entries = list(self._window)
        return entries[-k:]

    def write(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._window.append(content)

    def reset(self) -> None:
        self._window.clear()

    @property
    def memory_type(self) -> str:
        return "window_buffer"
