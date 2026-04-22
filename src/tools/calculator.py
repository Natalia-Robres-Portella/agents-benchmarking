"""CalculatorTool — sandboxed arithmetic evaluator.

Security: only numeric tokens and basic operators are permitted.
The expression is evaluated in a builtins-free namespace so no
Python built-ins or imports are accessible to the expression string.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar, Dict

from src.schema import ToolResult
from src.tools.base import Tool

# Allowlist: digits, spaces, arithmetic operators, dot, parens, %, **
_SAFE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%\^]+$")


class CalculatorTool(Tool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate a mathematical expression and return the result as a string. "
        "Supports +, -, *, /, %, ** and parentheses."
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '(3 + 4) * 2'.",
            }
        },
        "required": ["expression"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        expr: str = str(kwargs.get("expression", "")).strip()
        # Replace ^ with ** for user convenience
        expr = expr.replace("^", "**")
        if not _SAFE.match(expr):
            return ToolResult(
                output="",
                error=f"unsafe expression rejected: {expr!r}",
            )
        try:
            result = eval(expr, {"__builtins__": {}})  # noqa: S307
            return ToolResult(output=str(result))
        except ZeroDivisionError:
            return ToolResult(output="", error="division by zero")
        except Exception as exc:
            return ToolResult(output="", error=f"evaluation error: {exc}")
