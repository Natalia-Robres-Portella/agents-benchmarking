"""Search tools — MockSearchTool (deterministic) and LiveSearchTool (DuckDuckGo).

Benchmark runs MUST use MockSearchTool (seeded fixture, reproducible).
LiveSearchTool is for qualitative experiments only and is never counted
in official results — following WebArena's reproducibility principle
(Zhou et al., 2023).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from src.schema import ToolResult
from src.tools.base import Tool

_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent / "fixtures" / "search_responses.json"
)


class MockSearchTool(Tool):
    """
    Deterministic search backed by a JSON fixture.
    Keys are normalised (lower-case, stripped) so minor query variations match.
    """

    name: ClassVar[str] = "search"
    description: ClassVar[str] = (
        "Search for information about a topic. "
        "Returns a short text snippet relevant to your query."
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, fixture_path: Optional[str] = None, **_unused: Any) -> None:
        path = Path(fixture_path) if fixture_path else _DEFAULT_FIXTURE
        if path.exists():
            with open(path) as fh:
                raw: Dict[str, str] = json.load(fh)
            self._responses = {k.strip().lower(): v for k, v in raw.items()}
        else:
            self._responses = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = str(kwargs.get("query", "")).strip().lower()
        # Exact match first
        if query in self._responses:
            return ToolResult(output=self._responses[query])
        # Substring match fallback (first key that is contained in the query)
        for key, val in self._responses.items():
            if key in query or query in key:
                return ToolResult(output=val)
        return ToolResult(output=f"No results found for: {kwargs.get('query', '')}")


class LiveSearchTool(Tool):
    """
    DuckDuckGo search — non-deterministic, NOT for reproducible benchmarks.
    Used only for qualitative / exploratory experiments.
    """

    name: ClassVar[str] = "search"
    description: ClassVar[str] = MockSearchTool.description
    parameters: ClassVar[Dict[str, Any]] = MockSearchTool.parameters

    def __init__(self, max_results: int = 3) -> None:
        self._max_results = max_results

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]

            query = str(kwargs.get("query", ""))
            results = DDGS().text(query, max_results=self._max_results)
            snippets = [r.get("body", "") for r in results if r.get("body")]
            return ToolResult(output="\n\n".join(snippets) or "No results found.")
        except Exception as exc:
            return ToolResult(output="", error=f"search_error: {exc}")
