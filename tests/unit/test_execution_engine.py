"""Unit tests for ExecutionEngine — mock agent and trace logger."""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from src.agents.base import Agent
from src.config import ExperimentConfig
from src.execution_engine import ExecutionEngine
from src.schema import (
    Action,
    AgentState,
    Observation,
    Step,
    TaskInstance,
    Trajectory,
)
from src.tools.base import ToolRegistry
from src.tools.finish import FinishTool
from src.trace_logger import TraceLogger


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class FixedAgent(Agent):
    """Returns the same sequence of actions on each trial."""

    def __init__(self, actions: List[Action]) -> None:
        self._queue: List[Action] = []
        self._template = actions
        self._step = 0

    def act(self, state: AgentState) -> Action:
        idx = min(self._step, len(self._template) - 1)
        action = self._template[idx]
        self._step += 1
        return action

    def observe(self, obs: Observation) -> None:
        pass

    def reset(self, seed: int) -> None:
        self._step = 0

    @property
    def agent_id(self) -> str:
        return "fixed__no_memory__mock"

    # expose strategy for post_episode_hook
    class _FakeStrategy:
        def post_episode_hook(self, traj, agent): return None
    strategy = _FakeStrategy()


class SpyLogger(TraceLogger):
    """Records calls without writing to disk."""

    def __init__(self) -> None:
        self.opened: List[dict] = []
        self.steps: List[Step] = []
        self.closed: List[Trajectory] = []
        self._current_meta: dict = {}
        self._current_steps: List[Step] = []

    def open_trajectory(self, run_id, task_id, agent_id, trial_num, seed, config_hash) -> None:
        self._current_meta = dict(
            run_id=run_id, task_id=task_id, agent_id=agent_id,
            trial_num=trial_num, seed=seed, config_hash=config_hash,
        )
        self._current_steps = []
        self.opened.append(self._current_meta.copy())

    def log_step(self, step: Step) -> None:
        self._current_steps.append(step)
        self.steps.append(step)

    def close_trajectory(self, final_answer, termination) -> Trajectory:
        traj = Trajectory(
            **self._current_meta,
            steps=self._current_steps[:],
            termination=termination,
            final_answer=final_answer,
            total_tokens=sum(s.tokens_in + s.tokens_out for s in self._current_steps),
            total_latency_ms=sum(s.latency_ms for s in self._current_steps),
        )
        self.closed.append(traj)
        return traj

    def load_trajectories(self, run_dir: str) -> List[Trajectory]:
        return self.closed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task() -> TaskInstance:
    return TaskInstance(task_id="t1", input="What is 2+2?", gold="4")


@pytest.fixture
def cfg() -> ExperimentConfig:
    return ExperimentConfig(id="test", seed=42, n_trials=2, max_steps=5)


def _make_engine(agent: Agent, cfg: ExperimentConfig) -> tuple[ExecutionEngine, SpyLogger]:
    spy = SpyLogger()
    tools = ToolRegistry()
    tools.register(FinishTool())
    engine = ExecutionEngine(
        config=cfg,
        agent=agent,
        trace_logger=spy,
        tools=tools,
        run_id="run_test",
        config_hash="a" * 64,
    )
    return engine, spy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_direct_answer_terminates_immediately(task: TaskInstance, cfg: ExperimentConfig) -> None:
    agent = FixedAgent([Action(action_type="final_answer", final_answer="4", raw_llm_out="4")])
    engine, spy = _make_engine(agent, cfg)
    trajs = engine.run(task)
    assert len(trajs) == 2               # n_trials=2
    assert trajs[0].termination == "success"
    assert trajs[0].final_answer == "4"


def test_tool_call_then_finish(task: TaskInstance, cfg: ExperimentConfig) -> None:
    actions = [
        Action(
            action_type="tool_call",
            tool_name="finish",
            tool_args={"answer": "4"},
            raw_llm_out="",
        )
    ]
    agent = FixedAgent(actions)
    engine, spy = _make_engine(agent, cfg)
    trajs = engine.run(task)
    assert trajs[0].termination == "success"
    assert trajs[0].final_answer == "4"


def test_max_steps_termination(task: TaskInstance) -> None:
    cfg = ExperimentConfig(id="t", seed=0, n_trials=1, max_steps=3)
    agent = FixedAgent([
        Action(
            action_type="tool_call",
            tool_name="finish",
            tool_args={"answer": "x"},
            raw_llm_out="",
        )
    ])
    # Sabotage: give an action that never terminates
    agent._template = [
        Action(action_type="tool_call", tool_name="nonexistent", tool_args={}, raw_llm_out="")
    ]
    engine, spy = _make_engine(agent, cfg)
    trajs = engine.run(task)
    assert trajs[0].termination == "max_steps"


def test_abort_action_sets_parse_error(task: TaskInstance, cfg: ExperimentConfig) -> None:
    agent = FixedAgent([Action(action_type="abort", raw_llm_out="garbage")])
    engine, spy = _make_engine(agent, cfg)
    trajs = engine.run(task)
    assert trajs[0].termination == "parse_error"


def test_trajectories_logged_per_trial(task: TaskInstance, cfg: ExperimentConfig) -> None:
    agent = FixedAgent([Action(action_type="final_answer", final_answer="4", raw_llm_out="4")])
    engine, spy = _make_engine(agent, cfg)
    engine.run(task)
    assert len(spy.closed) == cfg.n_trials   # one trajectory per trial


def test_total_tokens_summed_correctly(task: TaskInstance) -> None:
    cfg = ExperimentConfig(id="t", seed=0, n_trials=1, max_steps=5)
    action = Action(
        action_type="final_answer", final_answer="4", raw_llm_out="4",
        tokens_in=10, tokens_out=5, token_count=15,
    )
    agent = FixedAgent([action])
    engine, spy = _make_engine(agent, cfg)
    trajs = engine.run(task)
    assert trajs[0].total_tokens == 15
