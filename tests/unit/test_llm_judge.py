"""Unit tests for LLMJudgeValidator — parser, bias mitigation, aggregation."""
from __future__ import annotations

import logging
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.evaluation.validators.llm_judge import (
    LLMJudgeValidator,
    _parse_score,
)
from src.llm.base import LLMResponse
from src.schema import TaskInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(task_id="t1", question="What is the capital of France?") -> TaskInstance:
    return TaskInstance(task_id=task_id, input=question, gold="Paris")


def _mock_llm(responses: List[str]) -> Any:
    """Returns a mock LLM that cycles through a list of string responses."""
    backend = MagicMock()
    backend.complete.side_effect = [
        LLMResponse(content=r, tokens_in=10, tokens_out=5, model="mock")
        for r in responses
    ]
    return backend


def _fixed_llm(score_text: str, n: int = 20) -> Any:
    """Mock LLM that always returns the same response, up to n times."""
    return _mock_llm([score_text] * n)


# ---------------------------------------------------------------------------
# _parse_score — score extraction robustness
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_simple_integer(self) -> None:
        assert _parse_score("SCORE: 8") == pytest.approx(0.8)

    def test_out_of_ten(self) -> None:
        assert _parse_score("SCORE: 7 out of 10") == pytest.approx(0.7)

    def test_slash_notation(self) -> None:
        assert _parse_score("SCORE: 9/10") == pytest.approx(0.9)

    def test_decimal_normalised(self) -> None:
        assert _parse_score("SCORE: 0.6") == pytest.approx(0.6)

    def test_bare_integer(self) -> None:
        assert _parse_score("10") == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert _parse_score("SCORE: 0") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        assert _parse_score("score: 5") == pytest.approx(0.5)

    def test_unparseable_returns_none(self) -> None:
        assert _parse_score("I cannot evaluate this.") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_score("") is None

    def test_clamps_above_ten(self) -> None:
        # Value > 10 is treated as being on the 0-10 scale
        result = _parse_score("SCORE: 10")
        assert result == pytest.approx(1.0)

    def test_score_with_dash(self) -> None:
        assert _parse_score("SCORE- 7") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# LLMJudgeValidator — basic scoring
# ---------------------------------------------------------------------------

class TestLLMJudgeValidator:
    def test_name(self) -> None:
        llm = _fixed_llm("SCORE: 8")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        assert v.name == "llm_judge"

    def test_perfect_score(self) -> None:
        llm = _fixed_llm("SCORE: 10")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("Paris", "Paris", _task())
        assert score == pytest.approx(1.0)

    def test_zero_score(self) -> None:
        llm = _fixed_llm("SCORE: 0")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("London", "Paris", _task())
        assert score == pytest.approx(0.0)

    def test_mid_range_score(self) -> None:
        llm = _fixed_llm("SCORE: 6")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("France", "Paris", _task())
        assert 0.0 < score < 1.0

    def test_returns_mean_across_templates(self) -> None:
        # 4 templates × 1 sample = 4 calls; alternates 8 and 4 → mean = 0.6
        responses = ["SCORE: 8", "SCORE: 4"] * 4
        llm = _mock_llm(responses)
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("Paris", "Paris", _task())
        assert score == pytest.approx(0.6)

    def test_n_samples_multiplies_calls(self) -> None:
        llm = _fixed_llm("SCORE: 5")
        v = LLMJudgeValidator(llm=llm, n_samples=2)
        v.validate("Paris", "Paris", _task())
        # 4 templates × 2 samples = 8 calls
        assert llm.complete.call_count == 8

    def test_single_sample_mode(self) -> None:
        llm = _fixed_llm("SCORE: 7")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        v.validate("Paris", "Paris", _task())
        assert llm.complete.call_count == 4   # 4 templates × 1 sample


# ---------------------------------------------------------------------------
# Robustness — LLM failures and bad outputs
# ---------------------------------------------------------------------------

class TestLLMJudgeRobustness:
    def test_all_calls_fail_returns_zero(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API down")
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("Paris", "Paris", _task())
        assert score == pytest.approx(0.0)

    def test_unparseable_responses_excluded(self) -> None:
        # 2 parseable (SCORE: 10), 2 unparseable — mean of parseable = 1.0
        responses = ["SCORE: 10", "cannot evaluate", "SCORE: 10", "no score here"]
        llm = _mock_llm(responses)
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("Paris", "Paris", _task())
        assert score == pytest.approx(1.0)

    def test_partial_failures_use_successful_calls(self) -> None:
        # 3 successful (SCORE: 8), 1 raises exception
        responses = ["SCORE: 8", "SCORE: 8", "SCORE: 8"]
        llm = MagicMock()
        llm.complete.side_effect = (
            [LLMResponse(content=r, tokens_in=1, tokens_out=1, model="m") for r in responses]
            + [RuntimeError("timeout")]
        )
        v = LLMJudgeValidator(llm=llm, n_samples=1)
        score = v.validate("Paris", "Paris", _task())
        assert score == pytest.approx(0.8)

    def test_low_confidence_warning_logged(self, caplog) -> None:
        # High variance: alternating 0 and 10 → std = 0.5, well above threshold
        responses = ["SCORE: 0", "SCORE: 10"] * 4
        llm = _mock_llm(responses)
        v = LLMJudgeValidator(llm=llm, n_samples=1, confidence_threshold=0.15)
        with caplog.at_level(logging.WARNING, logger="src.evaluation.validators.llm_judge"):
            v.validate("Paris", "Paris", _task())
        assert any("LOW CONFIDENCE" in r.message for r in caplog.records)

    def test_high_agreement_no_warning(self, caplog) -> None:
        llm = _fixed_llm("SCORE: 8")
        v = LLMJudgeValidator(llm=llm, n_samples=2, confidence_threshold=0.15)
        with caplog.at_level(logging.WARNING, logger="src.evaluation.validators.llm_judge"):
            v.validate("Paris", "Paris", _task())
        assert not any("LOW CONFIDENCE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Positional debiasing — 4 templates × 2 orderings wired correctly
# ---------------------------------------------------------------------------

class TestPositionalDebiasing:
    def test_four_different_prompts_sent(self) -> None:
        """Each of the 4 templates must produce a distinct prompt text."""
        prompts_seen: List[str] = []

        class CaptureLLM:
            def complete(self, prompt: str, **kwargs) -> LLMResponse:
                prompts_seen.append(prompt)
                return LLMResponse(content="SCORE: 5", tokens_in=1, tokens_out=1, model="m")

        v = LLMJudgeValidator(llm=CaptureLLM(), n_samples=1)
        v.validate("Paris", "The capital of France", _task())

        assert len(prompts_seen) == 4, f"Expected 4 prompts, got {len(prompts_seen)}"
        # All prompts must be distinct (different ordering / wording)
        assert len(set(prompts_seen)) == 4, "Templates must produce distinct prompts"

    def test_prediction_appears_before_gold_in_some_prompts(self) -> None:
        prompts_seen: List[str] = []

        class CaptureLLM:
            def complete(self, prompt: str, **kwargs) -> LLMResponse:
                prompts_seen.append(prompt)
                return LLMResponse(content="SCORE: 5", tokens_in=1, tokens_out=1, model="m")

        v = LLMJudgeValidator(llm=CaptureLLM(), n_samples=1)
        v.validate("PRED_TEXT", "GOLD_TEXT", _task())

        pred_first = sum(
            1 for p in prompts_seen
            if p.index("PRED_TEXT") < p.index("GOLD_TEXT")
        )
        gold_first = sum(
            1 for p in prompts_seen
            if p.index("GOLD_TEXT") < p.index("PRED_TEXT")
        )
        # Must have both orderings represented
        assert pred_first >= 1, "No template puts prediction first"
        assert gold_first >= 1, "No template puts gold first"


# ---------------------------------------------------------------------------
# Integration with EvaluationModule
# ---------------------------------------------------------------------------

def test_evaluation_module_with_llm_judge() -> None:
    """EvaluationModule wires LLMJudgeValidator when validator='llm_judge'."""
    from src.config import EvaluationConfig
    from src.evaluation.module import EvaluationModule
    from src.schema import Trajectory

    llm = _fixed_llm("SCORE: 10")
    cfg = EvaluationConfig(
        metrics=["success_rate"],
        validator="llm_judge",
        pass_k_values=[1],
    )
    module = EvaluationModule(cfg, llm_backend=llm)
    assert module._validator.name == "llm_judge"

    task = _task()
    traj = Trajectory(
        run_id="r", task_id="t1", agent_id="a",
        trial_num=0, seed=0, config_hash="x" * 64,
        final_answer="Paris", termination="success",
    )
    module.compute_all([traj], [task])
    assert traj.score == pytest.approx(1.0)
    assert traj.success is True
