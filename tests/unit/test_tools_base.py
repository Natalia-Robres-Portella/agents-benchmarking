"""Tests for ToolRegistry — validation, execution, error handling."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

import pytest

from src.schema import ToolResult
from src.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Minimal stub tools for testing
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Returns the input text."
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."},
        },
        "required": ["text"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=kwargs["text"])


class BrokenTool(Tool):
    name: ClassVar[str] = "broken"
    description: ClassVar[str] = "Always raises."
    parameters: ClassVar[Dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("internal failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(EchoTool())
    r.register(BrokenTool())
    return r


def test_register_and_get(registry: ToolRegistry) -> None:
    tool = registry.get("echo")
    assert tool.name == "echo"


def test_get_unknown_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get("nonexistent")


def test_valid_execution(registry: ToolRegistry) -> None:
    result = registry.validate_and_execute("echo", {"text": "hello"})
    assert result.output == "hello"
    assert result.error is None
    assert result.arg_valid is True


def test_missing_required_arg(registry: ToolRegistry) -> None:
    result = registry.validate_and_execute("echo", {})
    assert result.arg_valid is False
    assert "arg_validation_failed" in (result.error or "")
    assert result.output == ""


def test_unknown_tool_name(registry: ToolRegistry) -> None:
    result = registry.validate_and_execute("ghost", {"x": 1})
    assert result.arg_valid is False
    assert "unknown_tool" in (result.error or "")


def test_tool_execution_error_is_captured(registry: ToolRegistry) -> None:
    result = registry.validate_and_execute("broken", {})
    assert result.arg_valid is True             # args were valid
    assert "tool_execution_error" in (result.error or "")


def test_format_for_prompt_contains_tool_name(registry: ToolRegistry) -> None:
    prompt_block = registry.format_for_prompt()
    assert "echo" in prompt_block
    assert "text" in prompt_block


def test_list_tools(registry: ToolRegistry) -> None:
    names = [t.name for t in registry.list_tools()]
    assert "echo" in names
    assert "broken" in names
