"""StepCountMetric — mean steps per episode, breakdown by outcome."""
from __future__ import annotations

from typing import Dict, List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class StepCountMetric(Metric):
    """
    Mean number of ReAct steps taken per trajectory.

    Primary value: mean steps across all trajectories.
    breakdown:
      "success" — mean steps for trajectories that succeeded
      "failure" — mean steps for trajectories that did not succeed
    """

    @property
    def name(self) -> str:
        return "step_count"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        if not trajectories:
            return MetricResult(name=self.name, value=0.0)

        all_steps = [len(t.steps) for t in trajectories]
        mean_steps = sum(all_steps) / len(all_steps)

        success_steps = [len(t.steps) for t in trajectories if t.success]
        failure_steps = [len(t.steps) for t in trajectories if not t.success]

        breakdown: Dict[str, float] = {}
        if success_steps:
            breakdown["success"] = sum(success_steps) / len(success_steps)
        if failure_steps:
            breakdown["failure"] = sum(failure_steps) / len(failure_steps)

        return MetricResult(name=self.name, value=mean_steps, breakdown=breakdown)
