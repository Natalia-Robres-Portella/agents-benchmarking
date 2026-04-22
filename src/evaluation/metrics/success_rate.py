"""SuccessRateMetric — fraction of tasks where at least one trial succeeded."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class SuccessRateMetric(Metric):
    """
    For each task, take the max success across all trials.
    Average that over all tasks.

    Breakdown by task metadata["level"] when available.
    """

    @property
    def name(self) -> str:
        return "success_rate"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        if not trajectories:
            return MetricResult(name=self.name, value=0.0)

        task_map: Dict[str, TaskInstance] = {t.task_id: t for t in tasks}

        # Best-success per task (across trials)
        best: Dict[str, float] = defaultdict(float)
        for traj in trajectories:
            best[traj.task_id] = max(best[traj.task_id], float(traj.success))

        value = sum(best.values()) / len(best)

        # Breakdown by difficulty level
        by_level: Dict[str, List[float]] = defaultdict(list)
        for tid, score in best.items():
            level = task_map.get(tid, TaskInstance(task_id=tid, input="", gold="")).metadata.get(
                "level", "unknown"
            )
            by_level[level].append(score)

        breakdown = {lvl: sum(v) / len(v) for lvl, v in by_level.items()}
        return MetricResult(name=self.name, value=value, breakdown=breakdown)
