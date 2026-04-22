"""Unit tests for BaseAgent — mock LLM backend, real strategy + memory."""
from __future__ import annotations

import pytest

from src.agents.base import BaseAgent
from src.config import LLMConfig, MemoryConfig
from src.llm.base import LLMBackend, LLMResponse
from src.memory.factory import build_memory
from src.schema import Action, AgentState, Observation, TaskInstance, Trajectory
from src.strategies.direct import DirectAnswerStrategy
from src.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Minimal mock LLM
# ---------------------------------------------------------------------------

class MockLLM(LLMBackend):
    def __init__(self, content: str = "Paris") -> None:
        self._content = content

    def complete(self, prompt: str, stop=None, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=self._content,
            tokens_in=30,
            tokens_out=5,
            model="mock-model",
            latency_ms=10.0,
        )

    @property
    def model_id(self) -> str:
        return "mock-model"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task() -> TaskInstance:
    return TaskInstance(
        task_id="t1",
        input="What is the capital of France?",
        gold="Paris",
    )


@pytest.fixture
def observation(task: TaskInstance) -> Observation:
    return Observation(content=task.input, source="environment")


@pytest.fixture
def agent() -> BaseAgent:
    return BaseAgent(
        strategy=DirectAnswerStrategy(),
        memory=build_memory(MemoryConfig(type="no_memory")),
        llm=MockLLM("Paris"),
        tools=ToolRegistry(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_agent_id_format(agent: BaseAgent) -> None:
    assert agent.agent_id == "direct__no_memory__mock-model"


def test_act_returns_final_answer(
    agent: BaseAgent, task: TaskInstance, observation: Observation
) -> None:
    state = AgentState(task=task, observation=observation)
    action = agent.act(state)
    assert action.action_type == "final_answer"
    assert action.final_answer == "Paris"


def test_act_populates_token_counts(
    agent: BaseAgent, task: TaskInstance, observation: Observation
) -> None:
    state = AgentState(task=task, observation=observation)
    action = agent.act(state)
    assert action.tokens_in == 30
    assert action.tokens_out == 5
    assert action.token_count == 35


def test_act_abort_on_empty_response(
    task: TaskInstance, observation: Observation
) -> None:
    agent = BaseAgent(
        strategy=DirectAnswerStrategy(),
        memory=build_memory(MemoryConfig(type="no_memory")),
        llm=MockLLM(""),          # empty response → parse fails → abort
        tools=ToolRegistry(),
    )
    state = AgentState(task=task, observation=observation)
    action = agent.act(state)
    assert action.action_type == "abort"


def test_observe_writes_to_window_memory() -> None:
    agent = BaseAgent(
        strategy=DirectAnswerStrategy(),
        memory=build_memory(MemoryConfig(type="window_buffer", window_size=5)),
        llm=MockLLM(),
        tools=ToolRegistry(),
    )
    obs = Observation(content="Paris is in France.", source="search")
    agent.observe(obs)
    context = agent.memory_read("France")
    assert len(context) == 1
    assert "Paris" in context[0]


def test_reset_clears_window_memory() -> None:
    agent = BaseAgent(
        strategy=DirectAnswerStrategy(),
        memory=build_memory(MemoryConfig(type="window_buffer", window_size=5)),
        llm=MockLLM(),
        tools=ToolRegistry(),
    )
    agent.observe(Observation(content="something", source="env"))
    assert agent.memory_read("") != []
    agent.reset(seed=0)
    assert agent.memory_read("") == []


def test_reflect_returns_none_for_direct_strategy(agent: BaseAgent) -> None:
    traj = Trajectory(
        run_id="r", task_id="t", agent_id="a",
        trial_num=0, seed=0, config_hash="x" * 64,
    )
    assert agent.reflect(traj) is None
