"""
EvaluationModule — two-stage evaluation pipeline.

Stage 1 — Scoring:
  TaskValidator.validate(prediction, gold, task) → float [0, 1]
  Sets trajectory.score and trajectory.success on each Trajectory object.

Stage 2 — Metrics:
  Metric.compute(trajectories, tasks) → MetricResult
  Computes each configured metric over the full set of scored trajectories.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.config import EvaluationConfig
from src.evaluation.metrics import METRIC_REGISTRY
from src.evaluation.metrics.base import Metric
from src.evaluation.metrics.pass_at_k import PassAtKMetric
from src.evaluation.validators import VALIDATOR_REGISTRY
from src.schema import MetricResult, TaskInstance, Trajectory
from src.tasks.base import TaskValidator

logger = logging.getLogger(__name__)


class EvaluationModule:
    """
    Reads trajectories, scores them via the configured TaskValidator,
    then applies all registered Metric objects.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        llm_backend: Optional[object] = None,
    ) -> None:
        # Import concrete validators/metrics so they register themselves
        import src.evaluation.validators  # noqa: F401
        import src.evaluation.metrics     # noqa: F401

        if config.validator == "llm_judge":
            # LLMJudgeValidator requires an LLM backend — cannot use the
            # registry's no-arg constructor.
            from src.evaluation.validators.llm_judge import LLMJudgeValidator
            if llm_backend is None:
                # Build a default backend from the judge model config.
                from src.config import LLMConfig
                from src.llm.factory import build_llm_backend
                judge_model = config.llm_judge_model or "gpt-4o"
                llm_cfg = LLMConfig(
                    provider="openai",
                    model=judge_model,
                    temperature=0.3,
                )
                llm_backend = build_llm_backend(llm_cfg)
            self._validator: TaskValidator = LLMJudgeValidator(llm=llm_backend)
        else:
            self._validator = VALIDATOR_REGISTRY.get(config.validator)

        self._metrics: List[Metric] = []
        for name in config.metrics:
            if name == "pass_at_k":
                self._metrics.append(PassAtKMetric(k_values=config.pass_k_values))
            else:
                try:
                    self._metrics.append(METRIC_REGISTRY.get(name))
                except KeyError:
                    logger.warning("Unknown metric %r — skipping.", name)

    def score_trajectory(
        self, trajectory: Trajectory, task: TaskInstance
    ) -> float:
        return self._validator.validate(
            trajectory.final_answer or "", task.gold, task
        )

    def compute_all(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> Dict[str, MetricResult]:
        task_map: Dict[str, TaskInstance] = {t.task_id: t for t in tasks}

        # Stage 1: score each trajectory in-place
        for traj in trajectories:
            if traj.task_id not in task_map:
                continue
            traj.score = self.score_trajectory(traj, task_map[traj.task_id])
            traj.success = traj.score >= 1.0

        # Stage 2: compute metrics
        return {
            metric.name: metric.compute(trajectories, tasks)
            for metric in self._metrics
        }
