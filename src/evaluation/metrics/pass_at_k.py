"""
PassAtKMetric — unbiased estimator of pass@k.

Uses the closed-form formula from Reflexion (Shinn et al., NeurIPS 2023):
  pass@k = 1 - C(n-c, k) / C(n, k)
where n = total trials per task, c = successful trials, k = target k.

This is the same estimator used by OpenAI for HumanEval evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Dict, List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k random samples is correct."""
    if n < k:
        return float(c > 0)  # fewer trials than k — use empirical success
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


class PassAtKMetric(Metric):
    """
    Computes pass@k for each k value and averages across tasks.
    The primary `value` is pass@1.
    `breakdown` contains {"pass@1": …, "pass@3": …, "pass@5": …}.
    """

    def __init__(self, k_values: List[int] | None = None) -> None:
        self._k_values = k_values or [1, 3, 5]

    @property
    def name(self) -> str:
        return "pass_at_k"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        if not trajectories:
            return MetricResult(name=self.name, value=0.0)

        groups: Dict[str, List[Trajectory]] = defaultdict(list)
        for t in trajectories:
            groups[t.task_id].append(t)

        breakdown: Dict[str, float] = {}
        for k in self._k_values:
            per_task = []
            for trajs in groups.values():
                n = len(trajs)
                c = sum(t.success for t in trajs)
                per_task.append(_pass_at_k(n, c, k))
            breakdown[f"pass@{k}"] = sum(per_task) / len(per_task) if per_task else 0.0

        value = breakdown.get("pass@1", 0.0)
        return MetricResult(name=self.name, value=value, breakdown=breakdown)
