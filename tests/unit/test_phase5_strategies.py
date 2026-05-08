"""Unit tests for Reflexion, PlanAndExecute, and TreeOfThoughts strategies."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.schema import Action, AgentState, Observation, Step, TaskInstance, Trajectory
from src.strategies.plan_execute import PlanAndExecuteStrategy
from src.strategies.reflexion import ReflexionStrategy
from src.strategies.tot import TreeOfThoughtsStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task() -> TaskInstance:
    return TaskInstance(task_id="t1", input="Who founded Apple?", gold="Steve Jobs")


@pytest.fixture
def state(task: TaskInstance) -> AgentState:
    obs = Observation(content=task.input, source="environment")
    return AgentState(task=task, observation=obs)


@pytest.fixture
def state_with_history(task: TaskInstance) -> AgentState:
    first_raw = (
        "Plan:\n1. Search for Apple founder\n2. Confirm and answer\n\n"
        "Thought: I'll search now.\nAction: search\nAction Input: {\"query\": \"Apple founder\"}"
    )
    step = Step(
        step_id=0,
        thought="I'll search now.",
        action=Action(
            action_type="tool_call",
            tool_name="search",
            tool_args={"query": "Apple founder"},
            raw_llm_out=first_raw,
        ),
        observation=Observation(content="Apple was founded by Steve Jobs.", source="search"),
    )
    obs = Observation(content="Apple was founded by Steve Jobs.", source="search")
    return AgentState(task=task, observation=obs, history=[step])


def make_failed_trajectory(task_id: str = "t1") -> Trajectory:
    return Trajectory(
        run_id="r1",
        task_id=task_id,
        agent_id="reflexion__episodic__gpt-4o",
        trial_num=0,
        seed=42,
        config_hash="a" * 64,
        steps=[],
        termination="max_steps",
        success=False,
        score=0.0,
    )


# ---------------------------------------------------------------------------
# ReflexionStrategy
# ---------------------------------------------------------------------------

class TestReflexionStrategy:

    @pytest.fixture
    def strategy(self) -> ReflexionStrategy:
        return ReflexionStrategy()

    def test_name(self, strategy: ReflexionStrategy) -> None:
        assert strategy.name == "reflexion"

    def test_stop_sequences_inherited(self, strategy: ReflexionStrategy) -> None:
        assert "\nObservation:" in strategy.stop_sequences

    def test_build_prompt_contains_question(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        prompt = strategy.build_prompt(state, [], "search: finds info")
        assert "Who founded Apple?" in prompt

    def test_build_prompt_injects_memory(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        prompt = strategy.build_prompt(
            state, ["Last time I forgot to verify the year."], ""
        )
        assert "Last time I forgot to verify the year." in prompt

    def test_parse_tool_call(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        raw = (
            "Thought: I should search.\nAction: search\n"
            'Action Input: {"query": "Apple founder"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "tool_call"
        assert action.tool_name == "search"

    def test_parse_finish(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        raw = (
            "Thought: I know the answer.\nAction: finish\n"
            'Action Input: {"answer": "Steve Jobs"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "final_answer"
        assert action.final_answer == "Steve Jobs"

    def test_post_episode_hook_skipped_on_success(
        self, strategy: ReflexionStrategy
    ) -> None:
        traj = make_failed_trajectory()
        traj.success = True
        agent = MagicMock()
        result = strategy.post_episode_hook(traj, agent)
        assert result is None
        agent.llm.complete.assert_not_called()

    def test_post_episode_hook_writes_reflection_on_failure(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        # Prime the last question
        strategy.build_prompt(state, [], "")

        traj = make_failed_trajectory()
        mock_resp = MagicMock()
        mock_resp.content = "I should have searched more specifically."
        agent = MagicMock()
        agent.llm.complete.return_value = mock_resp

        reflection = strategy.post_episode_hook(traj, agent)

        assert reflection == "I should have searched more specifically."
        agent.memory_write.assert_called_once()

    def test_post_episode_hook_handles_llm_error(
        self, strategy: ReflexionStrategy, state: AgentState
    ) -> None:
        strategy.build_prompt(state, [], "")
        traj = make_failed_trajectory()
        agent = MagicMock()
        agent.llm.complete.side_effect = RuntimeError("timeout")

        result = strategy.post_episode_hook(traj, agent)
        assert result is None  # must not raise


# ---------------------------------------------------------------------------
# PlanAndExecuteStrategy
# ---------------------------------------------------------------------------

class TestPlanAndExecuteStrategy:

    @pytest.fixture
    def strategy(self) -> PlanAndExecuteStrategy:
        return PlanAndExecuteStrategy()

    def test_name(self, strategy: PlanAndExecuteStrategy) -> None:
        assert strategy.name == "plan_execute"

    def test_stop_sequences(self, strategy: PlanAndExecuteStrategy) -> None:
        assert "\nObservation:" in strategy.stop_sequences

    def test_build_prompt_no_history(
        self, strategy: PlanAndExecuteStrategy, state: AgentState
    ) -> None:
        prompt = strategy.build_prompt(state, [], "search: finds info")
        assert "Who founded Apple?" in prompt
        assert "PHASE 1" in prompt

    def test_build_prompt_with_history_shows_plan(
        self, strategy: PlanAndExecuteStrategy, state_with_history: AgentState
    ) -> None:
        prompt = strategy.build_prompt(state_with_history, [], "search: finds info")
        assert "Search for Apple founder" in prompt
        assert "✓" in prompt  # step 0 should be checked

    def test_parse_tool_call(
        self, strategy: PlanAndExecuteStrategy, state: AgentState
    ) -> None:
        raw = (
            "Plan:\n1. Search\n2. Answer\n\n"
            "Thought: Searching.\nAction: search\n"
            'Action Input: {"query": "Apple"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "tool_call"
        assert action.tool_name == "search"

    def test_parse_finish(
        self, strategy: PlanAndExecuteStrategy, state: AgentState
    ) -> None:
        raw = (
            "Thought: Done.\nAction: finish\n"
            'Action Input: {"answer": "Steve Jobs"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "final_answer"
        assert action.final_answer == "Steve Jobs"

    def test_parse_missing_action_returns_abort(
        self, strategy: PlanAndExecuteStrategy, state: AgentState
    ) -> None:
        action = strategy.parse_response("Plan:\n1. Do something", state)
        assert action.action_type == "abort"

    def test_annotate_plan_marks_completed(
        self, strategy: PlanAndExecuteStrategy
    ) -> None:
        plan = "1. Search\n2. Verify\n3. Answer"
        annotated = strategy._annotate_plan(plan, completed_steps=1)
        assert "✓" in annotated
        assert "← current" in annotated


# ---------------------------------------------------------------------------
# TreeOfThoughtsStrategy
# ---------------------------------------------------------------------------

class TestTreeOfThoughtsStrategy:

    @pytest.fixture
    def strategy(self) -> TreeOfThoughtsStrategy:
        return TreeOfThoughtsStrategy()

    def test_name(self, strategy: TreeOfThoughtsStrategy) -> None:
        assert strategy.name == "tot"

    def test_stop_sequences(self, strategy: TreeOfThoughtsStrategy) -> None:
        assert "\nObservation:" in strategy.stop_sequences

    def test_build_prompt_contains_question(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        prompt = strategy.build_prompt(state, [], "search: finds info")
        assert "Who founded Apple?" in prompt
        assert "Candidates:" in prompt

    def test_build_prompt_injects_memory(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        prompt = strategy.build_prompt(state, ["Avoid broad searches."], "")
        assert "Avoid broad searches." in prompt

    def test_parse_full_tot_response(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        raw = (
            "Candidates:\n"
            "1. Search Wikipedia for Apple Inc\n"
            "2. Search for Steve Jobs\n"
            "3. Search for Apple history\n\n"
            "Evaluation:\n"
            "1. Most direct\n2. Also good\n3. Too broad\n\n"
            "Best: 1\n"
            "Thought: I should search Wikipedia for Apple Inc.\n"
            "Action: search\n"
            'Action Input: {"query": "Apple Inc founder"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "tool_call"
        assert action.tool_name == "search"
        assert "Wikipedia" in (action.thought or "")

    def test_parse_finish(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        raw = (
            "Best: 2\n"
            "Thought: I now know the answer.\n"
            "Action: finish\n"
            'Action Input: {"answer": "Steve Jobs"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "final_answer"
        assert action.final_answer == "Steve Jobs"

    def test_parse_missing_action_returns_abort(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        action = strategy.parse_response("Candidates:\n1. Think\n2. Act\n3. Both", state)
        assert action.action_type == "abort"

    def test_parse_fallback_thought_without_best(
        self, strategy: TreeOfThoughtsStrategy, state: AgentState
    ) -> None:
        raw = (
            "Thought: Just searching directly.\n"
            "Action: search\n"
            'Action Input: {"query": "Apple"}'
        )
        action = strategy.parse_response(raw, state)
        assert action.action_type == "tool_call"
