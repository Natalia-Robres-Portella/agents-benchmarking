"""Unit tests for Phase 5 metrics: StepCount, ToolAccuracy, FailureRecovery, Latency."""
from __future__ import annotations

import pytest

from src.evaluation.metrics.failure_recovery import FailureRecoveryMetric
from src.evaluation.metrics.latency import LatencyMetric
from src.evaluation.metrics.steps import StepCountMetric
from src.evaluation.metrics.tool_accuracy import ToolAccuracyMetric
from src.schema import Action, Observation, Step, Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(action_type="tool_call", tool_name="search") -> Action:
    return Action(
        action_type=action_type,
        tool_name=tool_name,
        tool_args={"query": "x"},
        raw_llm_out="",
    )


def _obs() -> Observation:
    return Observation(content="result", source="search")


def _step(step_id=0, action_type="tool_call", tool_name="search",
          arg_valid=True, tool_error=None) -> Step:
    return Step(
        step_id=step_id,
        action=_action(action_type=action_type, tool_name=tool_name),
        observation=_obs(),
        arg_valid=arg_valid,
        tool_error=tool_error,
    )


def _traj(task_id="t1", success=True, steps=None, latency_ms=0.0,
          trial=0) -> Trajectory:
    return Trajectory(
        run_id="r", task_id=task_id, agent_id="a",
        trial_num=trial, seed=0, config_hash="x" * 64,
        steps=steps or [],
        success=success,
        total_latency_ms=latency_ms,
        termination="success" if success else "max_steps",
    )


# ---------------------------------------------------------------------------
# StepCountMetric
# ---------------------------------------------------------------------------

class TestStepCountMetric:
    m = StepCountMetric()

    def test_name(self) -> None:
        assert self.m.name == "step_count"

    def test_empty_trajectories(self) -> None:
        result = self.m.compute([], [])
        assert result.value == 0.0

    def test_mean_steps(self) -> None:
        trajs = [
            _traj(steps=[_step(0), _step(1)]),   # 2 steps
            _traj(steps=[_step(0)]),              # 1 step
        ]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(1.5)

    def test_breakdown_success_failure(self) -> None:
        trajs = [
            _traj(success=True, steps=[_step(0), _step(1)]),    # 2 steps, success
            _traj(success=False, steps=[_step(0), _step(1), _step(2)]),  # 3 steps, fail
        ]
        result = self.m.compute(trajs, [])
        assert result.breakdown["success"] == pytest.approx(2.0)
        assert result.breakdown["failure"] == pytest.approx(3.0)

    def test_only_successes(self) -> None:
        trajs = [_traj(success=True, steps=[_step(0)])]
        result = self.m.compute(trajs, [])
        assert "failure" not in result.breakdown

    def test_only_failures(self) -> None:
        trajs = [_traj(success=False, steps=[_step(0)])]
        result = self.m.compute(trajs, [])
        assert "success" not in result.breakdown

    def test_single_step_each(self) -> None:
        trajs = [_traj(steps=[_step()]), _traj(steps=[_step()])]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ToolAccuracyMetric
# ---------------------------------------------------------------------------

class TestToolAccuracyMetric:
    m = ToolAccuracyMetric()

    def test_name(self) -> None:
        assert self.m.name == "tool_accuracy"

    def test_no_tool_calls(self) -> None:
        trajs = [_traj(steps=[_step(action_type="final_answer")])]
        result = self.m.compute(trajs, [])
        assert result.value == 0.0

    def test_all_valid(self) -> None:
        trajs = [_traj(steps=[_step(arg_valid=True), _step(arg_valid=True)])]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(1.0)
        assert result.breakdown["arg_valid_rate"] == pytest.approx(1.0)
        assert result.breakdown["error_rate"] == pytest.approx(0.0)

    def test_half_valid(self) -> None:
        trajs = [_traj(steps=[
            _step(0, arg_valid=True),
            _step(1, arg_valid=False),
        ])]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(0.5)

    def test_error_rate(self) -> None:
        trajs = [_traj(steps=[
            _step(0, tool_error="failed"),
            _step(1, tool_error=None),
        ])]
        result = self.m.compute(trajs, [])
        assert result.breakdown["error_rate"] == pytest.approx(0.5)

    def test_calls_per_episode(self) -> None:
        trajs = [
            _traj(steps=[_step(0), _step(1)]),
            _traj(steps=[_step(0)]),
        ]
        result = self.m.compute(trajs, [])
        assert result.breakdown["calls_per_episode"] == pytest.approx(1.5)

    def test_per_tool_breakdown(self) -> None:
        trajs = [_traj(steps=[
            _step(0, tool_name="search", arg_valid=True),
            _step(1, tool_name="search", arg_valid=False),
            _step(2, tool_name="calculator", arg_valid=True),
        ])]
        result = self.m.compute(trajs, [])
        assert result.breakdown["tool.search.arg_valid_rate"] == pytest.approx(0.5)
        assert result.breakdown["tool.calculator.arg_valid_rate"] == pytest.approx(1.0)

    def test_empty_trajectories(self) -> None:
        result = self.m.compute([], [])
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# FailureRecoveryMetric
# ---------------------------------------------------------------------------

class TestFailureRecoveryMetric:
    m = FailureRecoveryMetric()

    def test_name(self) -> None:
        assert self.m.name == "failure_recovery"

    def test_no_errors_anywhere(self) -> None:
        trajs = [_traj(success=True, steps=[_step()])]
        result = self.m.compute(trajs, [])
        assert result.value == 0.0
        assert result.breakdown["episodes_with_errors"] == 0.0

    def test_error_and_recovery(self) -> None:
        trajs = [_traj(success=True, steps=[
            _step(0, tool_error="timeout"),
            _step(1),
        ])]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(1.0)
        assert result.breakdown["recovered"] == 1.0

    def test_error_no_recovery(self) -> None:
        trajs = [_traj(success=False, steps=[
            _step(0, tool_error="404"),
        ])]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(0.0)
        assert result.breakdown["recovered"] == 0.0

    def test_partial_recovery(self) -> None:
        trajs = [
            _traj("t1", success=True, steps=[_step(0, tool_error="err")]),   # recovered
            _traj("t2", success=False, steps=[_step(0, tool_error="err")]),  # not recovered
        ]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(0.5)
        assert result.breakdown["episodes_with_errors"] == 2.0
        assert result.breakdown["recovered"] == 1.0

    def test_mixed_with_clean_episode(self) -> None:
        # One episode has an error but recovers; one has no errors at all.
        trajs = [
            _traj("t1", success=True, steps=[_step(0, tool_error="err")]),
            _traj("t2", success=True, steps=[_step()]),  # no errors
        ]
        result = self.m.compute(trajs, [])
        assert result.breakdown["episodes_with_errors"] == 1.0
        assert result.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LatencyMetric
# ---------------------------------------------------------------------------

class TestLatencyMetric:
    m = LatencyMetric()

    def test_name(self) -> None:
        assert self.m.name == "latency"

    def test_empty_trajectories(self) -> None:
        result = self.m.compute([], [])
        assert result.value == 0.0

    def test_single_trajectory(self) -> None:
        trajs = [_traj(latency_ms=200.0)]
        result = self.m.compute(trajs, [])
        assert result.value == pytest.approx(200.0)
        assert result.breakdown["p50"] == pytest.approx(200.0)
        assert result.breakdown["p95"] == pytest.approx(200.0)
        assert result.breakdown["mean"] == pytest.approx(200.0)

    def test_p50_median(self) -> None:
        # 5 values: 100, 200, 300, 400, 500 — median (P50) = 200 (nearest-rank, 0-indexed → idx=1)
        trajs = [_traj(latency_ms=v) for v in [300, 100, 500, 200, 400]]
        result = self.m.compute(trajs, [])
        # P50: rank = int(0.5 * 5) - 1 = 1 → sorted[1] = 200
        assert result.breakdown["p50"] == pytest.approx(200.0)

    def test_p95_tail(self) -> None:
        trajs = [_traj(latency_ms=float(i * 100)) for i in range(1, 21)]  # 100..2000
        result = self.m.compute(trajs, [])
        # P95: rank = int(0.95*20)-1 = 18 → sorted[18] = 1900
        assert result.breakdown["p95"] == pytest.approx(1900.0)

    def test_mean(self) -> None:
        trajs = [_traj(latency_ms=v) for v in [100.0, 200.0, 300.0]]
        result = self.m.compute(trajs, [])
        assert result.breakdown["mean"] == pytest.approx(200.0)

    def test_primary_value_is_p50(self) -> None:
        trajs = [_traj(latency_ms=v) for v in [100.0, 200.0, 300.0]]
        result = self.m.compute(trajs, [])
        assert result.value == result.breakdown["p50"]
