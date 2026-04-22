"""No-op memory: stateless baseline — every read returns nothing."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.memory.base import MemoryModule


class NoMemory(MemoryModule):
    """Stateless agent — no memory reads, no memory writes."""

    def read(self, query: str, k: int = 5) -> List[str]:
        return []

    def write(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass

    def reset(self) -> None:
        pass

    @property
    def memory_type(self) -> str:
        return "no_memory"
