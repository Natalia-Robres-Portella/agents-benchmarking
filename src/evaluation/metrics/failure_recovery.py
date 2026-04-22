"""FailureRecoveryMetric — fraction of error-containing episodes that recover."""
from __future__ import annotations

from typing import List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class FailureRecoveryMetric(Metric):
    """
    Measures an agent's ability to recover from tool errors within an episode.

    An episode "has errors" when any step contains a non-None tool_error.
    Recovery rate = fraction of error-containing episodes that ultimately succeed.

    A high value here indicates the agent can diagnose errors and try alternative
    approaches (a key capability measured in Reflexion and ReAct evaluations).

    Primary value: recovery_rate.
    breakdown:
      "episodes_with_errors" — count of error-containing trajectories
      "recovered"            — count that succeeded despite errors
    """

    @property
    def name(self) -> str:
        return "failure_recovery"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        error_episodes = [t for t in trajectories if any(s.tool_error for s in t.steps)]

        if not error_episodes:
            return MetricResult(
                name=self.name,
                value=0.0,
                breakdown={"episodes_with_errors": 0.0, "recovered": 0.0},
            )

        recovered = sum(1 for t in error_episodes if t.success)
        rate = recovered / len(error_episodes)

        return MetricResult(
            name=self.name,
            value=rate,
            breakdown={
                "episodes_with_errors": float(len(error_episodes)),
                "recovered": float(recovered),
            },
        )
