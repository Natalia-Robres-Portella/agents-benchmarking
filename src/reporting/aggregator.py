"""ResultAggregator — groups trajectories by task_id for per-task analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.schema import Trajectory


class ResultAggregator:
    """Groups trajectories by task_id → {task_id: [trial_0, trial_1, …]}."""

    def aggregate(
        self, trajectories: List[Trajectory]
    ) -> Dict[str, List[Trajectory]]:
        groups: Dict[str, List[Trajectory]] = defaultdict(list)
        for t in trajectories:
            groups[t.task_id].append(t)
        return dict(groups)
