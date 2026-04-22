"""
Agent contract and BaseAgent concrete implementation.

Agent is the public interface used by ExecutionEngine.
BaseAgent wires a PlanningStrategy + MemoryModule + LLMBackend + ToolRegistry.

Design note: act() and observe() are separate methods (not a single step())
so that observe() can trigger memory writes independently of action generation.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from src.schema import Action, AgentState, Observation, Trajectory

if TYPE_CHECKING:
    from src.llm.base import LLMBackend
    from src.memory.base import MemoryModule
    from src.strategies.base import PlanningStrategy
    from src.tools.base import ToolRegistry


class Agent(ABC):

    @abstractmethod
    def act(self, state: AgentState) -> Action:
        """Given the current state, return the next action."""
        ...

    @abstractmethod
    def observe(self, obs: Observation) -> None:
        """Receive an observation; memory-enabled agents write to memory here."""
        ...

    @abstractmethod
    def reset(self, seed: int) -> None:
        """
        Reset for a new trial.
        Must: clear history, reset memory, re-seed any stochastic components.
        """
        ...

    def memory_read(self, query: str, k: int = 5) -> List[str]:
        """Optional: retrieve memories relevant to `query`.  No-op by default."""
        return []

    def memory_write(
        self, content: str, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        """Optional: persist content to memory.  No-op by default."""
        pass

    def reflect(self, trajectory: Trajectory) -> Optional[str]:
        """
        Optional: generate a verbal reflection after an episode.
        Reflexion overrides this; all other strategies return None.
        """
        return None

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """
        Human-readable fingerprint encoding strategy + memory + model.
        Format: "{strategy}__{memory_type}__{model_id}"
        Example: "react__no_memory__gpt-4o"
        """
        ...


# ---------------------------------------------------------------------------
# BaseAgent — concrete wiring of all components
# ---------------------------------------------------------------------------

class BaseAgent(Agent):
    """
    Wires PlanningStrategy + MemoryModule + LLMBackend + ToolRegistry.

    act() is the hot path: build prompt → call LLM → parse action.
    observe() writes the latest observation to memory.
    reset() is called by ExecutionEngine before every trial.
    """

    def __init__(
        self,
        strategy: "PlanningStrategy",
        memory: "MemoryModule",
        llm: "LLMBackend",
        tools: "ToolRegistry",
    ) -> None:
        self.strategy = strategy
        self.memory = memory
        self.llm = llm
        self.tools = tools

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, state: AgentState) -> Action:
        memory_context = self.memory.read(state.observation.content)
        tool_descriptions = self.tools.format_for_prompt()
        prompt = self.strategy.build_prompt(state, memory_context, tool_descriptions)
        llm_resp = self.llm.complete(prompt, stop=self.strategy.stop_sequences or None)
        action = self.strategy.parse_response(llm_resp.content, state)
        # Attach token counts so ExecutionEngine can populate Step correctly
        action.tokens_in = llm_resp.tokens_in
        action.tokens_out = llm_resp.tokens_out
        action.token_count = llm_resp.tokens_in + llm_resp.tokens_out
        return action

    def observe(self, obs: Observation) -> None:
        self.memory.write(obs.content, metadata={"source": obs.source})

    def reset(self, seed: int) -> None:
        self.memory.reset()
        random.seed(seed)

    # ------------------------------------------------------------------
    # Memory helpers (expose memory module through the Agent interface)
    # ------------------------------------------------------------------

    def memory_read(self, query: str, k: int = 5) -> List[str]:
        return self.memory.read(query, k)

    def memory_write(
        self, content: str, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        self.memory.write(content, metadata)  # type: ignore[arg-type]

    def reflect(self, trajectory: Trajectory) -> Optional[str]:
        return self.strategy.post_episode_hook(trajectory, self)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return (
            f"{self.strategy.name}"
            f"__{self.memory.memory_type}"
            f"__{self.llm.model_id}"
        )
