"""LatencyMetric — P50, P95, and mean of total episode latency."""
from __future__ import annotations

from typing import List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class LatencyMetric(Metric):
    """
    Percentile latency over all trajectories (total_latency_ms field).

    Primary value: P50 (median latency in milliseconds).
    breakdown: {"p50": ..., "p95": ..., "mean": ...}

    P95 is the key tail-latency indicator used in WebArena-style evaluations.
    """

    @property
    def name(self) -> str:
        return "latency"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        if not trajectories:
            return MetricResult(name=self.name, value=0.0)

        latencies = sorted(t.total_latency_ms for t in trajectories)
        n = len(latencies)

        def _percentile(pct: float) -> float:
            # nearest-rank method — consistent with numpy percentile(interpolation='lower')
            rank = max(0, int(pct / 100.0 * n) - 1)
            return latencies[min(rank, n - 1)]

        p50 = _percentile(50)
        p95 = _percentile(95)
        mean = sum(latencies) / n

        return MetricResult(
            name=self.name,
            value=p50,
            breakdown={"p50": p50, "p95": p95, "mean": mean},
        )
