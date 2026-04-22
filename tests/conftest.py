"""Shared pytest fixtures."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from src.schema import (
    Action,
    AgentState,
    Observation,
    Step,
    TaskInstance,
    Trajectory,
)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        experiment:
          id: "test_experiment"
          seed: 42
          n_trials: 2
          max_steps: 5
        agent:
          strategy: "react"
          llm:
            model: "gpt-4o"
        tasks:
          dataset: "hotpotqa"
          n_samples: 5
    """)
    p = tmp_path / "test_config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task() -> TaskInstance:
    return TaskInstance(
        task_id="task_001",
        input="What is the capital of France?",
        gold="Paris",
        metadata={"dataset": "test", "difficulty": "easy"},
    )


@pytest.fixture
def sample_observation() -> Observation:
    return Observation(content="France is a country in Western Europe.", source="search")


@pytest.fixture
def sample_action() -> Action:
    return Action(
        action_type="tool_call",
        tool_name="search",
        tool_args={"query": "capital of France"},
        thought="I need to look up the capital.",
        raw_llm_out="Thought: I need to look up the capital.\nAction: search\nAction Input: capital of France",
        token_count=42,
    )


@pytest.fixture
def sample_step(sample_action: Action, sample_observation: Observation) -> Step:
    return Step(
        step_id=0,
        thought=sample_action.thought,
        action=sample_action,
        observation=sample_observation,
        tokens_in=30,
        tokens_out=12,
        latency_ms=350.0,
        arg_valid=True,
    )


@pytest.fixture
def sample_trajectory(sample_step: Step, sample_task: TaskInstance) -> Trajectory:
    return Trajectory(
        run_id="run_abc123",
        task_id=sample_task.task_id,
        agent_id="react__no_memory__gpt-4o",
        trial_num=0,
        seed=42,
        config_hash="a" * 64,
        steps=[sample_step],
        termination="success",
        final_answer="Paris",
        total_tokens=42,
        total_latency_ms=350.0,
        success=True,
        score=1.0,
    )
