"""Tests for src/schema.py — all data models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schema import (
    Action,
    AgentState,
    MetricResult,
    Observation,
    Step,
    StepResult,
    TaskInstance,
    ToolResult,
    Trajectory,
)


class TestToolResult:
    def test_defaults(self) -> None:
        r = ToolResult(output="hello")
        assert r.error is None
        assert r.arg_valid is True
        assert r.metadata == {}

    def test_error_result(self) -> None:
        r = ToolResult(output="", error="tool_execution_error: timeout", arg_valid=False)
        assert r.arg_valid is False


class TestObservation:
    def test_defaults(self) -> None:
        obs = Observation(content="Paris is the capital.", source="search")
        assert obs.is_terminal is False
        assert obs.error is None

    def test_terminal(self) -> None:
        obs = Observation(content="Done", source="finish", is_terminal=True)
        assert obs.is_terminal is True


class TestAction:
    def test_tool_call(self, sample_action: Action) -> None:
        assert sample_action.action_type == "tool_call"
        assert sample_action.tool_name == "search"
        assert sample_action.tool_args == {"query": "capital of France"}

    def test_final_answer(self) -> None:
        a = Action(action_type="final_answer", final_answer="Paris", raw_llm_out="Paris")
        assert a.final_answer == "Paris"
        assert a.tool_name is None

    def test_invalid_action_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            Action(action_type="fly_to_moon", raw_llm_out="")  # type: ignore[arg-type]


class TestStep:
    def test_construction(self, sample_step: Step) -> None:
        assert sample_step.step_id == 0
        assert sample_step.arg_valid is True
        assert sample_step.latency_ms == 350.0


class TestTaskInstance:
    def test_required_fields(self) -> None:
        t = TaskInstance(task_id="t1", input="What?", gold="Answer")
        assert t.metadata == {}
        assert t.env_config is None

    def test_missing_task_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            TaskInstance(input="What?", gold="x")  # type: ignore[call-arg]


class TestAgentState:
    def test_empty_history(self, sample_task: TaskInstance, sample_observation: Observation) -> None:
        state = AgentState(task=sample_task, observation=sample_observation)
        assert state.history == []
        assert state.step_num == 0
        assert state.token_budget is None


class TestTrajectory:
    def test_defaults(self) -> None:
        t = Trajectory(
            run_id="r1", task_id="t1", agent_id="a1",
            trial_num=0, seed=42, config_hash="x" * 64,
        )
        assert t.steps == []
        assert t.success is False
        assert t.score == 0.0
        assert t.termination == "unknown"

    def test_full_trajectory(self, sample_trajectory: Trajectory) -> None:
        assert len(sample_trajectory.steps) == 1
        assert sample_trajectory.success is True
        assert sample_trajectory.final_answer == "Paris"


class TestMetricResult:
    def test_construction(self) -> None:
        m = MetricResult(name="success_rate", value=0.75)
        assert m.breakdown == {}
        assert m.metadata == {}
