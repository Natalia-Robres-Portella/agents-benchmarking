"""Memory module contract.  Three concrete variants: no_memory, window_buffer, vector_store."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryModule(ABC):
    """
    Pluggable memory backend.  Each agent holds one instance;
    reset() is called between trials to enforce isolation.
    """

    @abstractmethod
    def read(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve up to `k` relevant memory strings for the given query.
        Returns [] when memory is empty or not supported.
        """
        ...

    @abstractmethod
    def write(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a new memory entry."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories.  Called by agent.reset() between trials."""
        ...

    @property
    @abstractmethod
    def memory_type(self) -> str:
        """Identifier used in agent_id fingerprint (e.g. 'no_memory', 'window_buffer')."""
        ...
