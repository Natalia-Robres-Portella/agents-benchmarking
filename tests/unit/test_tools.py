"""Unit tests for concrete tools and the factory."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.schema import ToolResult
from src.tools.calculator import CalculatorTool
from src.tools.finish import FinishTool
from src.tools.search import MockSearchTool


# ---------------------------------------------------------------------------
# FinishTool
# ---------------------------------------------------------------------------

class TestFinishTool:
    def test_returns_answer(self) -> None:
        result = FinishTool().execute(answer="Paris")
        assert result.output == "Paris"
        assert result.error is None

    def test_sets_terminal_metadata(self) -> None:
        result = FinishTool().execute(answer="42")
        assert result.metadata.get("terminal") is True

    def test_missing_answer_caught_by_registry(self) -> None:
        from src.tools.base import ToolRegistry
        reg = ToolRegistry()
        reg.register(FinishTool())
        result = reg.validate_and_execute("finish", {})
        assert result.arg_valid is False


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def test_addition(self) -> None:
        assert CalculatorTool().execute(expression="2 + 3").output == "5"

    def test_multiplication(self) -> None:
        assert CalculatorTool().execute(expression="6 * 7").output == "42"

    def test_float(self) -> None:
        result = CalculatorTool().execute(expression="10 / 4")
        assert float(result.output) == pytest.approx(2.5)

    def test_exponent(self) -> None:
        assert CalculatorTool().execute(expression="2^10").output == "1024"

    def test_unsafe_expression_rejected(self) -> None:
        result = CalculatorTool().execute(expression="__import__('os').system('ls')")
        assert result.error is not None
        assert "unsafe" in result.error

    def test_division_by_zero(self) -> None:
        result = CalculatorTool().execute(expression="1/0")
        assert result.error is not None

    def test_empty_expression_rejected(self) -> None:
        result = CalculatorTool().execute(expression="")
        assert result.error is not None


# ---------------------------------------------------------------------------
# MockSearchTool
# ---------------------------------------------------------------------------

class TestMockSearchTool:
    def test_exact_match(self, tmp_path: Path) -> None:
        fixture = tmp_path / "responses.json"
        fixture.write_text(json.dumps({"capital of france": "Paris is the capital."}))
        tool = MockSearchTool(fixture_path=str(fixture))
        result = tool.execute(query="capital of france")
        assert "Paris" in result.output

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        fixture = tmp_path / "r.json"
        fixture.write_text(json.dumps({"eiffel tower": "Built in 1889."}))
        tool = MockSearchTool(fixture_path=str(fixture))
        result = tool.execute(query="Eiffel Tower")
        assert "1889" in result.output

    def test_no_match_returns_not_found(self, tmp_path: Path) -> None:
        fixture = tmp_path / "r.json"
        fixture.write_text(json.dumps({}))
        tool = MockSearchTool(fixture_path=str(fixture))
        result = tool.execute(query="quantum chromodynamics")
        assert "No results" in result.output

    def test_missing_fixture_returns_no_results(self) -> None:
        tool = MockSearchTool(fixture_path="/nonexistent/path.json")
        result = tool.execute(query="anything")
        assert "No results" in result.output
