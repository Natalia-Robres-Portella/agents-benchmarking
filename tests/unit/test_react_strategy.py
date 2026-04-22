"""Unit tests for ReactStrategy — prompt building and response parsing."""
from __future__ import annotations

import pytest

from src.schema import Action, AgentState, Observation, Step, TaskInstance
from src.strategies.react import ReactStrategy


@pytest.fixture
def strategy() -> ReactStrategy:
    return ReactStrategy()


@pytest.fixture
def state() -> AgentState:
    task = TaskInstance(task_id="t1", input="What is the capital of France?", gold="Paris")
    obs = Observation(content=task.input, source="environment")
    return AgentState(task=task, observation=obs)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def test_build_prompt_contains_question(strategy: ReactStrategy, state: AgentState) -> None:
    prompt = strategy.build_prompt(state, [], "search: finds information")
    assert "What is the capital of France?" in prompt


def test_build_prompt_contains_tool_descriptions(
    strategy: ReactStrategy, state: AgentState
) -> None:
    prompt = strategy.build_prompt(state, [], "search: finds information")
    assert "search: finds information" in prompt


def test_build_prompt_contains_memory_context(
    strategy: ReactStrategy, state: AgentState
) -> None:
    prompt = strategy.build_prompt(
        state,
        memory_context=["Last time I searched too broadly."],
        tool_descriptions="",
    )
    assert "Last time I searched too broadly." in prompt


def test_build_prompt_formats_history(strategy: ReactStrategy, state: AgentState) -> None:
    step = Step(
        step_id=0,
        thought="I need to search.",
        action=Action(
            action_type="tool_call",
            tool_name="search",
            tool_args={"query": "capital France"},
            raw_llm_out="",
        ),
        observation=Observation(content="Paris is the capital.", source="search"),
    )
    state_with_history = AgentState(
        task=state.task,
        observation=state.observation,
        history=[step],
    )
    prompt = strategy.build_prompt(state_with_history, [], "")
    assert "I need to search." in prompt
    assert "Paris is the capital." in prompt


# ---------------------------------------------------------------------------
# parse_response — tool call
# ---------------------------------------------------------------------------

def test_parse_tool_call(strategy: ReactStrategy, state: AgentState) -> None:
    raw = (
        "Thought: I should search for the capital.\n"
        "Action: search\n"
        'Action Input: {"query": "capital of France"}'
    )
    action = strategy.parse_response(raw, state)
    assert action.action_type == "tool_call"
    assert action.tool_name == "search"
    assert action.tool_args == {"query": "capital of France"}
    assert action.thought == "I should search for the capital."


# ---------------------------------------------------------------------------
# parse_response — finish / final answer
# ---------------------------------------------------------------------------

def test_parse_finish_action(strategy: ReactStrategy, state: AgentState) -> None:
    raw = (
        "Thought: I now know the final answer.\n"
        "Action: finish\n"
        'Action Input: {"answer": "Paris"}'
    )
    action = strategy.parse_response(raw, state)
    assert action.action_type == "final_answer"
    assert action.final_answer == "Paris"
    assert action.thought == "I now know the final answer."


def test_parse_finish_case_insensitive(strategy: ReactStrategy, state: AgentState) -> None:
    raw = (
        "Thought: Done.\nAction: Finish\nAction Input: {\"answer\": \"Paris\"}"
    )
    action = strategy.parse_response(raw, state)
    assert action.action_type == "final_answer"


# ---------------------------------------------------------------------------
# parse_response — abort on malformed output
# ---------------------------------------------------------------------------

def test_parse_missing_action_returns_abort(
    strategy: ReactStrategy, state: AgentState
) -> None:
    raw = "Thought: I'm confused."
    action = strategy.parse_response(raw, state)
    assert action.action_type == "abort"


def test_parse_empty_response_returns_abort(
    strategy: ReactStrategy, state: AgentState
) -> None:
    action = strategy.parse_response("", state)
    assert action.action_type == "abort"


def test_parse_invalid_json_args_still_calls_tool(
    strategy: ReactStrategy, state: AgentState
) -> None:
    raw = (
        "Thought: searching.\n"
        "Action: search\n"
        "Action Input: not valid json"
    )
    # Should fall back gracefully — tool_args may be None/empty
    action = strategy.parse_response(raw, state)
    # It could be abort (no valid args) or tool_call with empty args —
    # either is acceptable; the key invariant is no exception raised.
    assert action.action_type in ("tool_call", "abort")


# ---------------------------------------------------------------------------
# stop_sequences
# ---------------------------------------------------------------------------

def test_stop_sequences(strategy: ReactStrategy) -> None:
    assert "\nObservation:" in strategy.stop_sequences
