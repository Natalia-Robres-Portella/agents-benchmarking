"""
Plan-and-Execute: separates high-level planning from step-by-step execution.

Phase 1 (step 0, no history): the LLM produces a numbered plan and immediately
executes the first step.  Phase 2 (steps 1+): the plan is re-displayed with a
checkmark on completed steps so the model always knows where it is.

The plan is recovered from state.history[0].action.raw_llm_out each turn so
the strategy itself stays stateless between builds.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from src.schema import Action, AgentState, Step
from src.strategies.base import PlanningStrategy


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a planning agent that solves questions in two phases.

{tool_descriptions}

PHASE 1 — PLAN (only on your very first response):
Write a numbered plan of the concrete steps you will take to answer the question.
Then immediately execute step 1.

Use this exact format:

Plan:
1. <step one>
2. <step two>
...

Thought: <reasoning for step 1>
Action: <tool name>
Action Input: <valid JSON object>

PHASE 2 — EXECUTE (all subsequent responses):
You will be shown the plan with completed steps marked ✓.
Execute the next unchecked step using the same format:

Thought: <reasoning>
Action: <tool name>
Action Input: <valid JSON object>

To submit the final answer use:
Thought: I now have all the information needed.
Action: finish
Action Input: {{"answer": "<concise answer: name, year, or short phrase>"}}

Important: the answer must be a short extract, never a full sentence.

Begin!
"""

_QUESTION_BLOCK = "Question: {question}\n"

_PLAN_BLOCK = """\
Plan:
{annotated_plan}

"""

_HISTORY_STEP = (
    "Thought: {thought}\n"
    "Action: {action}\n"
    "Action Input: {args}\n"
    "Observation: {observation}\n"
)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class PlanAndExecuteStrategy(PlanningStrategy):

    def build_prompt(
        self,
        state: AgentState,
        memory_context: List[str],
        tool_descriptions: str,
    ) -> str:
        parts: List[str] = [_SYSTEM.format(tool_descriptions=tool_descriptions)]

        if memory_context:
            parts.append("Previous reflections (use these to avoid past mistakes):")
            for ref in memory_context:
                parts.append(f"- {ref}")
            parts.append("")

        parts.append(_QUESTION_BLOCK.format(question=state.task.input))

        if state.history:
            # Re-display the plan with checkmarks for completed steps.
            raw_plan = self._extract_plan(state.history[0].action.raw_llm_out)
            if raw_plan:
                annotated = self._annotate_plan(raw_plan, len(state.history))
                parts.append(_PLAN_BLOCK.format(annotated_plan=annotated))

            parts.append(self._format_history(state.history))

        return "\n".join(parts)

    def parse_response(self, raw: str, state: AgentState) -> Action:
        thought = self._extract_thought(raw)
        tool_name = self._extract_action(raw)
        tool_args = self._extract_action_input(raw)

        if tool_name is None:
            return Action(action_type="abort", thought=thought, raw_llm_out=raw)

        tool_name = tool_name.strip().lower()

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
        return "plan_execute"

    @property
    def stop_sequences(self) -> List[str]:
        return ["\nObservation:"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_plan(raw_llm_out: str) -> Optional[str]:
        """Pull the numbered list from the first LLM response."""
        m = re.search(
            r"Plan:\s*\n((?:\d+\..+\n?)+)",
            raw_llm_out,
            re.IGNORECASE,
        )
        return m.group(1).strip() if m else None

    @staticmethod
    def _annotate_plan(plan: str, completed_steps: int) -> str:
        """Mark the first `completed_steps` lines with ✓ and the next with ←."""
        lines = [l for l in plan.splitlines() if l.strip()]
        annotated: List[str] = []
        for i, line in enumerate(lines):
            if i < completed_steps:
                annotated.append(f"{line} ✓")
            elif i == completed_steps:
                annotated.append(f"{line}  ← current")
            else:
                annotated.append(line)
        return "\n".join(annotated)

    @staticmethod
    def _format_history(steps: List[Step]) -> str:
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
            line_m = re.search(r"Action Input:\s*(.+)", raw)
            if line_m:
                try:
                    return json.loads(line_m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        return None
