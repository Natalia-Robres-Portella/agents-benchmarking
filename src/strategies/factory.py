"""Build a PlanningStrategy from a strategy name string."""
from __future__ import annotations

from src.strategies.base import PlanningStrategy


def build_strategy(name: str) -> PlanningStrategy:
    if name == "direct":
        from src.strategies.direct import DirectAnswerStrategy
        return DirectAnswerStrategy()
    if name == "react":
        from src.strategies.react import ReactStrategy
        return ReactStrategy()
    if name == "reflexion":
        from src.strategies.reflexion import ReflexionStrategy  # Phase 5
        return ReflexionStrategy()
    if name == "plan_execute":
        from src.strategies.plan_execute import PlanAndExecuteStrategy  # Phase 5
        return PlanAndExecuteStrategy()
    if name == "tot":
        from src.strategies.tot import TreeOfThoughtsStrategy  # Phase 5
        return TreeOfThoughtsStrategy()
    raise ValueError(f"Unknown strategy: {name!r}")
