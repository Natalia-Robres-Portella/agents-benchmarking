"""DirectAnswerStrategy — the no-tool baseline.

Sends the question to the LLM verbatim and treats the entire response as
the final answer.  No reasoning trace.  No tool calls.  Used as the
lower-bound baseline in all benchmark comparisons.
"""
from __future__ import annotations

from typing import List

from src.schema import Action, AgentState
from src.strategies.base import PlanningStrategy

_PROMPT = """\
Answer the following question as concisely as possible.
Respond with the answer only — no explanation.

Question: {question}
"""


class DirectAnswerStrategy(PlanningStrategy):

    def build_prompt(
        self,
        state: AgentState,
        memory_context: List[str],
        tool_descriptions: str,
    ) -> str:
        return _PROMPT.format(question=state.task.input)

    def parse_response(self, raw: str, state: AgentState) -> Action:
        answer = raw.strip()
        if not answer:
            return Action(action_type="abort", raw_llm_out=raw)
        return Action(
            action_type="final_answer",
            final_answer=answer,
            raw_llm_out=raw,
        )

    @property
    def name(self) -> str:
        return "direct"
