"""FinishTool — signals successful episode termination with a final answer."""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from src.schema import ToolResult
from src.tools.base import Tool


class FinishTool(Tool):
    name: ClassVar[str] = "finish"
    description: ClassVar[str] = (
        "Call this tool when you have the final answer to the question. "
        "Pass your answer as the 'answer' argument."
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Your final answer to the question.",
            }
        },
        "required": ["answer"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            output=str(kwargs["answer"]),
            metadata={"terminal": True},
        )
