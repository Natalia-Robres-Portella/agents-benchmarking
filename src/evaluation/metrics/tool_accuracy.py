"""ToolAccuracyMetric — tool call validity rate and per-tool error breakdown."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.evaluation.metrics.base import Metric
from src.schema import MetricResult, TaskInstance, Trajectory


class ToolAccuracyMetric(Metric):
    """
    Measures the quality of tool use across all trajectories.

    A tool call is "valid" when Step.arg_valid is True (JSON-Schema check passed).
    A tool call has an "error" when Step.tool_error is not None.

    Primary value: arg_valid_rate (fraction of tool calls with valid arguments).
    breakdown:
      "arg_valid_rate"          — same as primary value
      "error_rate"              — fraction of calls that produced a tool error
      "calls_per_episode"       — mean tool calls per trajectory
      "tool.<name>.arg_valid_rate" — per-tool validity rate (one key per tool used)
    """

    @property
    def name(self) -> str:
        return "tool_accuracy"

    def compute(
        self,
        trajectories: List[Trajectory],
        tasks: List[TaskInstance],
    ) -> MetricResult:
        total_calls = 0
        valid_calls = 0
        error_calls = 0
        per_tool_total: Dict[str, int] = defaultdict(int)
        per_tool_valid: Dict[str, int] = defaultdict(int)

        for traj in trajectories:
            for step in traj.steps:
                if step.action.action_type != "tool_call":
                    continue
                total_calls += 1
                tool_name = step.action.tool_name or "unknown"
                per_tool_total[tool_name] += 1
                if step.arg_valid:
                    valid_calls += 1
                    per_tool_valid[tool_name] += 1
                if step.tool_error is not None:
                    error_calls += 1

        if total_calls == 0:
            return MetricResult(
                name=self.name,
                value=0.0,
                breakdown={"arg_valid_rate": 0.0, "error_rate": 0.0, "calls_per_episode": 0.0},
            )

        arg_valid_rate = valid_calls / total_calls
        error_rate = error_calls / total_calls
        calls_per_episode = total_calls / len(trajectories) if trajectories else 0.0

        breakdown: Dict[str, float] = {
            "arg_valid_rate": arg_valid_rate,
            "error_rate": error_rate,
            "calls_per_episode": calls_per_episode,
        }
        for tool, n in per_tool_total.items():
            breakdown[f"tool.{tool}.arg_valid_rate"] = per_tool_valid[tool] / n

        return MetricResult(name=self.name, value=arg_valid_rate, breakdown=breakdown)
