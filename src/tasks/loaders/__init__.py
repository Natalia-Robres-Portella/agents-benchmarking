"""Registers all task loaders with TASK_REGISTRY on import."""
from src.tasks.base import TASK_REGISTRY
from src.tasks.loaders.hotpotqa import HotPotQALoader

TASK_REGISTRY.register("hotpotqa", HotPotQALoader)

__all__ = ["HotPotQALoader"]
