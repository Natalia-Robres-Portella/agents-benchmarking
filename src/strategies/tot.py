"""
Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023).

This implementation fits ToT into a single-LLM-call-per-step architecture by asking
the model to explicitly generate N candidate thoughts, evaluate each one, select the
best, and then execute the corresponding action.  This captures the core insight of
ToT — explore and self-evaluate multiple reasoning paths — without requiring a
separate orchestration loop outside the strategy.
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
You are a deliberate reasoning agent that explores multiple lines of thought \
before acting.

{tool_descriptions}

At each step you MUST follow this exact format:

Candidates:
1. <first candidate thought about what to do next>
2. <second candidate thought about what to do next>
3. <third candidate thought about what to do next>

Evaluation:
1. <why candidate 1 is or isn't the best approach>
2. <why candidate 2 is or isn't the best approach>
3. <why candidate 3 is or isn't the best approach>

Best: <1, 2, or 3>
Thought: <the selected candidate thought, restated clearly>
Action: <one tool name from the list above>
Action Input: <valid JSON object matching that tool's parameters>

After each Action Input you will receive an Observation.  Then repeat the full \
Candidates / Evaluation / Best / Thought / Action / Action Input cycle.

When you have the final answer:
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

class TreeOfThoughtsStrategy(PlanningStrategy):

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
        return "tot"

    @property
    def stop_sequences(self) -> List[str]:
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
        # The authoritative Thought line follows "Best: N\n"
        m = re.search(r"Best:\s*\d+\s*\nThought:\s*(.+?)(?=\nAction:|\Z)", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: any Thought line
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
