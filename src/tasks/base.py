"""
Task contract: TaskLoader, TaskValidator, and their registries.

TaskInstance (defined in schema.py) carries no validator — it is pure data
and must be fully serialisable to JSONL.  Validators are stateless objects
selected from EvaluationConfig.validator, not stored on the task.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from src.schema import TaskInstance


# ---------------------------------------------------------------------------
# Validator contract
# ---------------------------------------------------------------------------

class TaskValidator(ABC):
    """
    Stateless scorer: given a prediction and ground truth, return [0, 1].
    Concrete implementations live in src/evaluation/validators/.
    """

    @abstractmethod
    def validate(
        self,
        prediction: Any,
        gold: Any,
        task: TaskInstance,
    ) -> float:
        """Return 1.0 for fully correct, 0.0 for fully wrong, in-between for partial."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------

class TaskLoader(ABC):
    """Loads a list of TaskInstance objects from a specific dataset."""

    @abstractmethod
    def load(
        self,
        split: str,
        n_samples: int,
        seed: int,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[TaskInstance]:
        """
        Return a deterministic list of `n_samples` tasks.
        Seeded shuffle ensures same tasks are used across all strategies.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TaskRegistry:
    """Maps dataset names to TaskLoader classes."""

    def __init__(self) -> None:
        self._loaders: Dict[str, Type[TaskLoader]] = {}

    def register(self, name: str, loader_cls: Type[TaskLoader]) -> None:
        self._loaders[name] = loader_cls

    def get(self, name: str) -> TaskLoader:
        if name not in self._loaders:
            raise KeyError(
                f"Dataset '{name}' not registered. Available: {list(self._loaders)}"
            )
        return self._loaders[name]()

    def list_datasets(self) -> List[str]:
        return list(self._loaders)


TASK_REGISTRY = TaskRegistry()
