"""TokensPerTaskMetric — mean total tokens consumed per trajectory."""
from __future__ import annotations

from typing import List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class TokensPerTaskMetric(Metric):

    @property
    def name(self) -> str:
        return "tokens_per_task"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        if not trajectories:
            return MetricResult(name=self.name, value=0.0)
        value = sum(t.total_tokens for t in trajectories) / len(trajectories)
        return MetricResult(name=self.name, value=value)
