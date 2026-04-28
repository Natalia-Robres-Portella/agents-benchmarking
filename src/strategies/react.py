"""
ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023).

The strategy interleaves Thought / Action / Observation triplets.  Each call to
build_prompt() formats the full history so the model can continue the trace.
parse_response() extracts the next Thought + Action from the raw LLM output.

Stop sequence "\nObservation:" prevents the model from hallucinating its own
observations — the engine supplies the real tool result instead.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, List, Optional

from src.schema import Action, AgentState, Step, Trajectory
from src.strategies.base import PlanningStrategy

if TYPE_CHECKING:
    from src.agents.base import Agent


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a reasoning agent that answers questions by thinking step by step \
and using tools.

{tool_descriptions}

Use this EXACT format for every response:
Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <valid JSON object matching that tool's parameters>

After each Action Input you will receive an Observation.  Then repeat with \
another Thought / Action / Action Input.

When you have the final answer call the finish tool:
Thought: I now know the final answer.
Action: finish
Action Input: {{"answer": "<concise answer: a name, year, place, or short phrase — NOT a full sentence>"}}

Important: the answer field must be a short extract, never a complete sentence.

Begin!
"""

_QUESTION_BLOCK = "Question: {question}\n"
_HISTORY_STEP = (
    "Thought: {thought}\n"
    "Action: {action}\n"
    "Action Input: {args}\n"
    "Observation: {observation}\n"
)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ReactStrategy(PlanningStrategy):

    def build_prompt(
        self,
        state: AgentState,
        memory_context: List[str],
        tool_descriptions: str,
    ) -> str:
        parts: List[str] = [_SYSTEM.format(tool_descriptions=tool_descriptions)]

        # Inject episodic reflections (Reflexion variant) above the question
        if memory_context:
            parts.append("Previous reflections (use these to avoid past mistakes):")
            for ref in memory_context:
                parts.append(f"- {ref}")
            parts.append("")

        parts.append(_QUESTION_BLOCK.format(question=state.task.input))
        parts.append(self._format_history(state.history))
        return "\n".join(parts)

    def parse_response(self, raw: str, state: AgentState) -> Action:
        """
        Extract Thought / Action / Action Input from the raw LLM output.
        Returns Action(action_type="abort") on any parse failure — never raises.
        """
        thought = self._extract_thought(raw)
        tool_name = self._extract_action(raw)
        tool_args = self._extract_action_input(raw)

        if tool_name is None:
            return Action(action_type="abort", thought=thought, raw_llm_out=raw)

        tool_name = tool_name.strip().lower()

        # finish tool → final_answer
        if tool_name == "finish":
            answer = (tool_args or {}).get("answer", raw.strip())
            return Action(
                action_type="final_answer",
                thought=thought,
                final_answer=str(answer),
                raw_llm_out=raw,
            )

        return Action(
            action_type="tool_call",
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args or {},
            raw_llm_out=raw,
        )

    @property
    def name(self) -> str:
        return "react"

    @property
    def stop_sequences(self) -> List[str]:
        # Stop before the model writes its own Observation
        return ["\nObservation:"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_history(steps: List[Step]) -> str:
        if not steps:
            return ""
        lines: List[str] = []
        for step in steps:
            thought = step.thought or ""
            tool_name = step.action.tool_name or ""
            args = json.dumps(step.action.tool_args or {})
            obs = step.observation.content
            lines.append(
                _HISTORY_STEP.format(
                    thought=thought,
                    action=tool_name,
                    args=args,
                    observation=obs,
                )
            )
        return "".join(lines)

    @staticmethod
    def _extract_thought(raw: str) -> Optional[str]:
        m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", raw, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_action(raw: str) -> Optional[str]:
        m = re.search(r"\nAction:\s*(.+?)(?=\n|$)", raw)
        if not m:
            # Try at start of string (first step, no leading newline)
            m = re.search(r"^Action:\s*(.+?)(?=\n|$)", raw, re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_action_input(raw: str) -> Optional[dict]:
        m = re.search(r"Action Input:\s*(\{.*?\})", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            # Best-effort: try to parse whatever is on that line
            line_m = re.search(r"Action Input:\s*(.+)", raw)
            if line_m:
                try:
                    return json.loads(line_m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        return None
