"""Episodic memory for Reflexion — accumulates verbal reflections across trials.

Reflexion (Shinn et al., 2023) keeps a per-task buffer of verbal self-critiques
so the agent can condition the next trial on past failures.  The key invariant:

  reset()      → no-op  (memory persists across trials on the SAME task)
  hard_reset() → clears (called by the orchestrator BETWEEN tasks)

This asymmetry is the core mechanism: reflections accumulate within a task,
are leveraged in the next trial, then cleared before a new task begins.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.memory.base import MemoryModule


class EpisodicMemory(MemoryModule):
    """Verbal reflection buffer used by ReflexionStrategy."""

    def __init__(self, max_reflections: int = 3) -> None:
        self._reflections: List[str] = []
        self._max = max_reflections

    def read(self, query: str, k: int = 5) -> List[str]:
        return self._reflections[-k:]

    def write(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._reflections.append(content)
        if len(self._reflections) > self._max:
            self._reflections = self._reflections[-self._max :]

    def reset(self) -> None:
        """Intentional no-op — reflections must survive trial boundaries."""

    def hard_reset(self) -> None:
        """Called between distinct tasks to prevent cross-task contamination."""
        self._reflections.clear()

    @property
    def memory_type(self) -> str:
        return "episodic"
