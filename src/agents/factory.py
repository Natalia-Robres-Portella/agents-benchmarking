"""Build a BaseAgent from an AgentConfig."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.config import AgentConfig
from src.llm.factory import build_llm_backend
from src.memory.factory import build_memory
from src.strategies.factory import build_strategy
from src.tools.factory import build_tool_registry


def build_agent(config: AgentConfig) -> BaseAgent:
    llm = build_llm_backend(config.llm)
    tools = build_tool_registry(config.tools)
    memory = build_memory(config.memory)
    strategy = build_strategy(config.strategy)
    return BaseAgent(strategy=strategy, memory=memory, llm=llm, tools=tools)
