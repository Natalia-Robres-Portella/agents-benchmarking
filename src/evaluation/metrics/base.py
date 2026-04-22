"""Metric contract and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

from src.schema import MetricResult, TaskInstance, Trajectory


class Metric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        """
        Compute this metric over all trajectories in one experiment run.
        `tasks` is aligned with trajectories by task_id for gold lookups.
        """
        ...


class MetricRegistry:
    """Maps metric names to Metric classes."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Type[Metric]] = {}

    def register(self, metric_cls: Type[Metric]) -> None:
        instance = metric_cls()
        self._metrics[instance.name] = metric_cls

    def get(self, name: str) -> Metric:
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not registered. Available: {list(self._metrics)}"
            )
        return self._metrics[name]()

    def list_metrics(self) -> List[str]:
        return list(self._metrics)


METRIC_REGISTRY = MetricRegistry()
