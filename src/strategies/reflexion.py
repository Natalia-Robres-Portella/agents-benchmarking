"""
Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023).

Extends ReAct with a verbal self-reflection loop.  After each failed trial the
agent generates a natural-language critique of what went wrong; that critique is
written to episodic memory so the next trial's prompt begins with a reminder of
past mistakes.  ReactStrategy already injects memory_context above the question,
so no prompt changes are needed here — only post_episode_hook is overridden.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from src.schema import AgentState, Trajectory
from src.strategies.react import ReactStrategy

if TYPE_CHECKING:
    from src.agents.base import Agent


_REFLECTION_PROMPT = """\
You are reflecting on a failed attempt to answer a question.

Question: {question}

Your trajectory:
{trajectory_summary}

Termination reason: {termination}

Write a concise reflection (2-4 sentences) covering:
- What went wrong or what was missing in your reasoning
- Any incorrect assumptions you made
- What you should do differently on the next attempt

Reflection:"""


class ReflexionStrategy(ReactStrategy):
    """ReAct + verbal reflection written to episodic memory after each failure."""

    def __init__(self) -> None:
        self._last_question: str = ""

    @property
    def name(self) -> str:
        return "reflexion"

    def build_prompt(
        self,
        state: AgentState,
        memory_context: List[str],
        tool_descriptions: str,
    ) -> str:
        # Track the question so post_episode_hook can reference it.
        self._last_question = state.task.input
        return super().build_prompt(state, memory_context, tool_descriptions)

    def post_episode_hook(
        self,
        trajectory: Trajectory,
        agent: "Agent",
    ) -> Optional[str]:
        if trajectory.success:
            return None

        lines: List[str] = []
        for step in trajectory.steps:
            thought = step.thought or "(no thought)"
            if step.action.action_type == "tool_call":
                lines.append(f"Thought: {thought}")
                lines.append(
                    f"Action: {step.action.tool_name}({step.action.tool_args})"
                )
                lines.append(f"Observation: {step.observation.content[:300]}")
            elif step.action.action_type == "final_answer":
                lines.append(f"Thought: {thought}")
                lines.append(f"Answer given: {step.action.final_answer}")

        trajectory_summary = "\n".join(lines) if lines else "(no steps recorded)"

        prompt = _REFLECTION_PROMPT.format(
            question=self._last_question or trajectory.task_id,
            trajectory_summary=trajectory_summary,
            termination=trajectory.termination,
        )

        try:
            resp = agent.llm.complete(prompt)
            reflection = resp.content.strip()
            if reflection:
                agent.memory_write(reflection, metadata={"type": "reflection"})
            return reflection or None
        except Exception:
            return None
