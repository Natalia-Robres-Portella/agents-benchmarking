"""Build a MemoryModule from a MemoryConfig."""
from __future__ import annotations

from src.config import MemoryConfig
from src.memory.base import MemoryModule
from src.memory.episodic import EpisodicMemory
from src.memory.no_memory import NoMemory
from src.memory.window_buffer import WindowBufferMemory


def build_memory(config: MemoryConfig) -> MemoryModule:
    if config.type == "no_memory":
        return NoMemory()
    if config.type == "window_buffer":
        return WindowBufferMemory(window_size=config.window_size)
    if config.type == "episodic":
        return EpisodicMemory()
    if config.type == "vector_store":
        from src.memory.vector_store import VectorStoreMemory  # optional dep
        return VectorStoreMemory(config.embedding_model, config.top_k)
    raise ValueError(f"Unknown memory type: {config.type!r}")
