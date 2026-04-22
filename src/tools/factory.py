"""Build a ToolRegistry from a list of ToolConfig objects."""
from __future__ import annotations

from typing import List

from src.config import ToolConfig
from src.tools.base import Tool, ToolRegistry

# Registry of available tool names → concrete Tool classes.
# Populated lazily so optional deps (search libs) are not imported on module load.
_TOOL_CLASSES: dict[str, type[Tool]] = {}


def _register_builtins() -> None:
    """Import and register built-in tools once."""
    if _TOOL_CLASSES:
        return
    from src.tools.calculator import CalculatorTool
    from src.tools.finish import FinishTool
    from src.tools.search import MockSearchTool

    _TOOL_CLASSES["finish"] = FinishTool
    _TOOL_CLASSES["calculator"] = CalculatorTool
    _TOOL_CLASSES["search"] = MockSearchTool


def build_tool_registry(tool_configs: List[ToolConfig]) -> ToolRegistry:
    """
    Instantiate and register each configured tool.

    During Step 2 (baseline agent), an empty config list is valid —
    the registry is empty, and DirectAnswerStrategy never calls any tool.
    """
    registry = ToolRegistry()
    if not tool_configs:
        return registry

    _register_builtins()
    for tc in tool_configs:
        name = tc.name
        if name not in _TOOL_CLASSES:
            raise ValueError(
                f"Unknown tool: {name!r}. Available: {list(_TOOL_CLASSES)}"
            )
        cls = _TOOL_CLASSES[name]
        instance = cls(**tc.config) if tc.config else cls()
        registry.register(instance)
    return registry
