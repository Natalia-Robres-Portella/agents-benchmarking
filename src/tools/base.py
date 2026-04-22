"""
Tool contract and registry.

Every tool execution — including argument validation — goes through
ToolRegistry.validate_and_execute(), never tool.execute() directly.
This ensures all tool errors are captured as structured ToolResult objects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Tuple

from src.schema import ToolResult


class Tool(ABC):
    """Abstract base for all tools.  Subclasses define name/description/parameters."""

    name: ClassVar[str]
    description: ClassVar[str]
    # JSON Schema for the tool's keyword arguments.
    # Used for: (1) argument validation, (2) prompt injection.
    parameters: ClassVar[Dict[str, Any]]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with validated arguments."""
        ...


class ToolRegistry:
    """Central registry; the only path through which tools are called."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not registered. Available: {list(self._tools)}"
            )
        return self._tools[name]

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def validate_and_execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Validate args against the tool's JSON Schema, then execute.
        Returns a ToolResult with error set (and arg_valid=False) on any failure.
        This is the ONLY call path for tool execution.
        """
        if name not in self._tools:
            return ToolResult(
                output="",
                error=f"unknown_tool: '{name}' not in registry",
                arg_valid=False,
                metadata={"tool_name": name},
            )

        tool = self._tools[name]
        valid, error_msg = self._validate_args(tool, args)

        if not valid:
            return ToolResult(
                output="",
                error=f"arg_validation_failed: {error_msg}",
                arg_valid=False,
                metadata={"tool_name": name, "provided_args": args},
            )

        try:
            return tool.execute(**args)
        except Exception as exc:
            return ToolResult(
                output="",
                error=f"tool_execution_error: {exc}",
                arg_valid=True,
                metadata={"tool_name": name},
            )

    def format_for_prompt(self) -> str:
        """Format all registered tools as a description block for agent prompts."""
        lines: List[str] = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(f"\n  {tool.name}: {tool.description}")
            props = tool.parameters.get("properties", {})
            required = set(tool.parameters.get("required", []))
            for param, spec in props.items():
                req_marker = " (required)" if param in required else ""
                desc = spec.get("description", "")
                lines.append(f"    - {param}{req_marker}: {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_args(tool: Tool, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Minimal required-field check.  Full JSON Schema validation is advanced scope."""
        required: List[str] = tool.parameters.get("required", [])
        for field in required:
            if field not in args:
                return False, f"missing required argument: '{field}'"
        return True, ""
