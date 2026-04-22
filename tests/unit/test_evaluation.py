"""Unit tests for validators, metrics, and EvaluationModule."""
from __future__ import annotations

from typing import List

import pytest

from src.config import EvaluationConfig
from src.evaluation.module import EvaluationModule
from src.evaluation.validators.exact_match import ExactMatchValidator
from src.evaluation.validators.fuzzy_match import FuzzyMatchValidator
from src.evaluation.metrics.pass_at_k import PassAtKMetric, _pass_at_k
from src.evaluation.metrics.success_rate import SuccessRateMetric
from src.evaluation.metrics.tokens import TokensPerTaskMetric
from src.schema import Action, Observation, Step, TaskInstance, Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _traj(task_id: str, answer: str, success: bool, score: float,
          trial: int = 0, tokens: int = 100) -> Trajectory:
    return Trajectory(
        run_id="r", task_id=task_id, agent_id="a",
        trial_num=trial, seed=42, config_hash="x" * 64,
        final_answer=answer, termination="success" if success else "max_steps",
        success=success, score=score, total_tokens=tokens,
    )


def _task(task_id: str, gold: str, level: str = "easy") -> TaskInstance:
    return TaskInstance(
        task_id=task_id, input="?", gold=gold,
        metadata={"level": level},
    )


# ---------------------------------------------------------------------------
# ExactMatchValidator
# ---------------------------------------------------------------------------

class TestExactMatchValidator:
    v = ExactMatchValidator()

    def test_exact_match(self) -> None:
        assert self.v.validate("Paris", "Paris", _task("t", "Paris")) == 1.0

    def test_case_insensitive(self) -> None:
        assert self.v.validate("paris", "PARIS", _task("t", "PARIS")) == 1.0

    def test_mismatch(self) -> None:
        assert self.v.validate("London", "Paris", _task("t", "Paris")) == 0.0

    def test_empty_prediction(self) -> None:
        assert self.v.validate("", "Paris", _task("t", "Paris")) == 0.0


# ---------------------------------------------------------------------------
# FuzzyMatchValidator
# ---------------------------------------------------------------------------

class TestFuzzyMatchValidator:
    v = FuzzyMatchValidator()

    def test_exact_match(self) -> None:
        assert self.v.validate("Paris", "Paris", _task("t", "Paris")) == 1.0

    def test_partial_match(self) -> None:
        score = self.v.validate("Paris France", "Paris", _task("t", "Paris"))
        assert 0.0 < score < 1.0

    def test_no_overlap(self) -> None:
        assert self.v.validate("London", "Paris", _task("t", "Paris")) == 0.0

    def test_both_empty(self) -> None:
        assert self.v.validate("", "", _task("t", "")) == 1.0

    def test_stop_words_ignored(self) -> None:
        # "the" is a stop word — "the capital" vs "capital" should match
        score = self.v.validate("the capital", "capital", _task("t", "capital"))
        assert score == 1.0


# ---------------------------------------------------------------------------
# PassAtKMetric — formula
# ---------------------------------------------------------------------------

class TestPassAtKFormula:
    def test_all_fail(self) -> None:
        assert _pass_at_k(5, 0, 1) == 0.0

    def test_all_pass(self) -> None:
        assert _pass_at_k(5, 5, 1) == 1.0

    def test_half_pass_k1(self) -> None:
        result = _pass_at_k(4, 2, 1)
        assert 0.0 < result < 1.0

    def test_fewer_trials_than_k(self) -> None:
        # Only 2 trials, asking for k=5 — falls back to empirical
        assert _pass_at_k(2, 1, 5) == 1.0


class TestPassAtKMetric:
    def test_computes_breakdown(self) -> None:
        trajs = [
            _traj("t1", "ok", True, 1.0, 0),
            _traj("t1", "ok", False, 0.0, 1),
            _traj("t1", "ok", False, 0.0, 2),
        ]
        metric = PassAtKMetric(k_values=[1, 3])
        result = metric.compute(trajs, [])
        assert "pass@1" in result.breakdown
        assert "pass@3" in result.breakdown

    def test_primary_value_is_pass_at_1(self) -> None:
        trajs = [_traj("t1", "ok", True, 1.0, 0)]
        result = PassAtKMetric(k_values=[1]).compute(trajs, [])
        assert result.value == result.breakdown["pass@1"]


# ---------------------------------------------------------------------------
# SuccessRateMetric
# ---------------------------------------------------------------------------

class TestSuccessRateMetric:
    def test_all_succeed(self) -> None:
        trajs = [_traj("t1", "x", True, 1.0), _traj("t2", "y", True, 1.0)]
        tasks = [_task("t1", "x"), _task("t2", "y")]
        result = SuccessRateMetric().compute(trajs, tasks)
        assert result.value == 1.0

    def test_none_succeed(self) -> None:
        trajs = [_traj("t1", "x", False, 0.0)]
        result = SuccessRateMetric().compute(trajs, [_task("t1", "x")])
        assert result.value == 0.0

    def test_best_across_trials(self) -> None:
        # t1 fails on trial 0 but succeeds on trial 1
        trajs = [
            _traj("t1", "x", False, 0.0, trial=0),
            _traj("t1", "x", True, 1.0, trial=1),
        ]
        result = SuccessRateMetric().compute(trajs, [_task("t1", "x")])
        assert result.value == 1.0

    def test_breakdown_by_level(self) -> None:
        trajs = [
            _traj("t1", "x", True, 1.0),
            _traj("t2", "y", False, 0.0),
        ]
        tasks = [_task("t1", "x", "easy"), _task("t2", "y", "hard")]
        result = SuccessRateMetric().compute(trajs, tasks)
        assert result.breakdown["easy"] == 1.0
        assert result.breakdown["hard"] == 0.0


# ---------------------------------------------------------------------------
# TokensPerTaskMetric
# ---------------------------------------------------------------------------

class TestTokensPerTaskMetric:
    def test_average_tokens(self) -> None:
        trajs = [_traj("t1", "x", True, 1.0, tokens=200), _traj("t2", "y", True, 1.0, tokens=100)]
        result = TokensPerTaskMetric().compute(trajs, [])
        assert result.value == pytest.approx(150.0)

    def test_empty_trajectories(self) -> None:
        result = TokensPerTaskMetric().compute([], [])
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# EvaluationModule integration
# ---------------------------------------------------------------------------

class TestEvaluationModule:
    def _cfg(self) -> EvaluationConfig:
        return EvaluationConfig(
            metrics=["success_rate", "pass_at_k", "tokens_per_task"],
            validator="fuzzy_match",
            pass_k_values=[1, 3],
        )

    def test_scores_and_computes_metrics(self) -> None:
        tasks = [_task("t1", "Paris"), _task("t2", "London")]
        trajs = [
            Trajectory(run_id="r", task_id="t1", agent_id="a", trial_num=0,
                       seed=42, config_hash="x" * 64, final_answer="Paris",
                       termination="success", total_tokens=50),
            Trajectory(run_id="r", task_id="t2", agent_id="a", trial_num=0,
                       seed=42, config_hash="x" * 64, final_answer="Berlin",
                       termination="success", total_tokens=80),
        ]
        module = EvaluationModule(self._cfg())
        metrics = module.compute_all(trajs, tasks)

        assert "success_rate" in metrics
        assert "pass_at_k" in metrics
        assert "tokens_per_task" in metrics
        # Paris matches Paris → success=True; Berlin≠London → success=False
        assert metrics["success_rate"].value == pytest.approx(0.5)

    def test_sets_score_on_trajectory(self) -> None:
        task = _task("t1", "Paris")
        traj = Trajectory(
            run_id="r", task_id="t1", agent_id="a", trial_num=0,
            seed=42, config_hash="x" * 64, final_answer="Paris",
            termination="success",
        )
        module = EvaluationModule(self._cfg())
        module.compute_all([traj], [task])
        assert traj.score == pytest.approx(1.0)
        assert traj.success is True
