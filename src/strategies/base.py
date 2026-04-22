"""
Planning strategy contract.

A strategy owns two things:
  1. How to build the prompt from the current agent state (build_prompt).
  2. How to parse the raw LLM response into a structured Action (parse_response).

Optionally it also owns what happens between episodes (post_episode_hook),
which Reflexion uses to generate verbal reflections.

Strategies are stateless — all episode state lives in Agent.history.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional  # noqa: F401

from src.schema import Action, AgentState, Trajectory

if TYPE_CHECKING:
    from src.agents.base import Agent


class PlanningStrategy(ABC):

    @abstractmethod
    def build_prompt(
        self,
        state: AgentState,
        memory_context: List[str],
        tool_descriptions: str,
    ) -> str:
        """
        Construct the full prompt string that will be sent to the LLM.

        Args:
            state:            current agent state (task + history + observation)
            memory_context:   strings returned by MemoryModule.read()
            tool_descriptions: formatted string from ToolRegistry.format_for_prompt()
        """
        ...

    @abstractmethod
    def parse_response(self, raw: str, state: AgentState) -> Action:
        """
        Parse the raw LLM output string into a structured Action.
        Must never raise — return Action(action_type="abort") on parse failure.
        """
        ...

    def post_episode_hook(
        self,
        trajectory: Trajectory,
        agent: "Agent",
    ) -> Optional[str]:
        """
        Called by the execution engine after each trial ends.
        Reflexion overrides this to generate a verbal reflection and write
        it to the agent's episodic memory buffer.
        Returns the reflection string, or None if not applicable.
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used in agent_id fingerprint (e.g. 'react', 'reflexion')."""
        ...

    @property
    def stop_sequences(self) -> List[str]:
        """Stop tokens passed to the LLM.  Override per strategy."""
        return []
