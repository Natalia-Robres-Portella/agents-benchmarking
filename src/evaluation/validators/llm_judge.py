"""
LLMJudgeValidator — bias-mitigated LLM-as-a-judge evaluator.

Bias mitigation strategy (Wang et al., 2023 — "Large Language Models are not
Robust Multiple Choice Selectors", arXiv:2309.03882; Zheng et al., 2023 —
"Judging LLM-as-a-Judge", arXiv:2306.05685):

  1. Prompt variation: two complementary templates (factual_precision and
     semantic_equivalence) evaluate orthogonal quality dimensions.

  2. Positional debiasing: each template is run twice — once with the
     prediction appearing before the gold, once with gold before prediction.
     Averaging across both orderings cancels first-position recency bias.

  3. Variance sampling: temperature > 0 induces stochastic responses.
     Running n_samples independent calls per (template, order) pair exposes
     judge instability — high std dev flags low-confidence scores.

  4. Score aggregation: mean over all 4 × n_samples calls.  When std dev
     exceeds `confidence_threshold`, a WARNING is emitted so the caller can
     decide whether to trust the score.

See docs/llm_judge_notes.md for a full discussion of limitations and mitigations.
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import Any, List, Optional, Tuple

from src.schema import TaskInstance
from src.tasks.base import TaskValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — 2 perspectives × 2 orderings = 4 prompts per call
# ---------------------------------------------------------------------------

# Each tuple is (prediction_field, gold_field) order:
#   "pred_first" → prediction appears before gold in the prompt body
#   "gold_first" → gold appears before prediction

_TEMPLATES: List[Tuple[str, str]] = [
    # ── Factual precision, prediction first ──────────────────────────────
    (
        "factual_pred_first",
        """\
You are a strict factual evaluator. Score how accurately the PREDICTION \
answers the question, compared to the GOLD ANSWER. \
Focus on factual correctness; ignore wording and style differences.

Question: {question}
Prediction: {prediction}
Gold Answer: {gold}

Score the PREDICTION from 0 to 10:
  10 = All key facts are present and correct
   5 = Some key facts correct but incomplete or partially wrong
   0 = Factually wrong or entirely unrelated

Respond with exactly one line: SCORE: <integer>""",
    ),
    # ── Factual precision, gold first ────────────────────────────────────
    (
        "factual_gold_first",
        """\
You are a strict factual evaluator. Score how accurately the PREDICTION \
answers the question, compared to the GOLD ANSWER. \
Focus on factual correctness; ignore wording and style differences.

Question: {question}
Gold Answer: {gold}
Prediction: {prediction}

Score the PREDICTION from 0 to 10:
  10 = All key facts are present and correct
   5 = Some key facts correct but incomplete or partially wrong
   0 = Factually wrong or entirely unrelated

Respond with exactly one line: SCORE: <integer>""",
    ),
    # ── Semantic equivalence, prediction first ───────────────────────────
    (
        "semantic_pred_first",
        """\
You are a semantic similarity evaluator. Score how semantically equivalent \
the PREDICTION is to the GOLD ANSWER, regardless of exact wording.

Question: {question}
Prediction: {prediction}
Gold Answer: {gold}

Score from 0 to 10:
  10 = Same meaning (paraphrase counts as 10)
   5 = Captures some but not all of the intended meaning
   0 = Unrelated or contradictory meaning

Respond with exactly one line: SCORE: <integer>""",
    ),
    # ── Semantic equivalence, gold first ─────────────────────────────────
    (
        "semantic_gold_first",
        """\
You are a semantic similarity evaluator. Score how semantically equivalent \
the PREDICTION is to the GOLD ANSWER, regardless of exact wording.

Question: {question}
Gold Answer: {gold}
Prediction: {prediction}

Score from 0 to 10:
  10 = Same meaning (paraphrase counts as 10)
   5 = Captures some but not all of the intended meaning
   0 = Unrelated or contradictory meaning

Respond with exactly one line: SCORE: <integer>""",
    ),
]

# Matches: "SCORE: 7", "7", "7/10", "7 out of 10", "score: 0.7"
_SCORE_RE = re.compile(
    r"""
    (?:SCORE\s*[:\-]\s*)?   # optional "SCORE:" prefix
    (\d+(?:\.\d+)?)         # numeric value (int or float)
    (?:\s*/\s*10)?          # optional /10 denominator
    (?:\s*out\s+of\s*10)?   # optional "out of 10"
    """,
    re.VERBOSE | re.IGNORECASE,
)

_LOW_CONFIDENCE_THRESHOLD = 0.15  # std dev above this flags the result


def _parse_score(text: str) -> Optional[float]:
    """
    Extract a normalised [0, 1] score from the judge's free-form response.
    Returns None when no numeric value can be found.
    """
    m = _SCORE_RE.search(text.strip())
    if not m:
        return None
    raw = float(m.group(1))
    # Scores > 1 are assumed to be on a 0–10 scale; normalise to [0, 1].
    if raw > 1.0:
        raw = min(raw, 10.0) / 10.0
    return raw


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class LLMJudgeValidator(TaskValidator):
    """
    Scores a (prediction, gold) pair using an LLM as judge.

    Bias is mitigated via four prompt templates (2 perspectives × 2 orderings)
    and n_samples independent calls per template with temperature > 0.

    The final score is the mean of all successful calls.  When the std dev
    across calls exceeds `confidence_threshold`, a WARNING is logged — the
    caller can inspect it via the Python logging system.

    Args:
        llm: any object with a `.complete(prompt, **kwargs) -> LLMResponse`
             method (i.e., an LLMBackend instance).  Typed as Any to avoid
             a circular import; the duck-typed interface is sufficient.
        n_samples: independent calls per (template, order) combination.
                   Total LLM calls = 4 × n_samples.  Default: 2.
        temperature: sampling temperature.  Must be > 0 to expose variance.
                     Default: 0.3.
        confidence_threshold: std dev threshold above which a WARNING is
                              logged.  Default: 0.15 (i.e., 1.5 points on
                              a 0–10 scale).
    """

    def __init__(
        self,
        llm: Any,
        n_samples: int = 2,
        temperature: float = 0.3,
        confidence_threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._llm = llm
        self._n_samples = max(1, n_samples)
        self._temperature = temperature
        self._conf_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "llm_judge"

    def validate(
        self,
        prediction: Any,
        gold: Any,
        task: TaskInstance,
    ) -> float:
        """
        Score `prediction` against `gold` with multi-template, multi-order,
        multi-sample judging.  Returns a float in [0, 1].
        """
        scores: List[float] = []

        for _template_name, template in _TEMPLATES:
            for _ in range(self._n_samples):
                prompt = template.format(
                    question=task.input,
                    prediction=str(prediction),
                    gold=str(gold),
                )
                try:
                    resp = self._llm.complete(prompt, temperature=self._temperature)
                    score = _parse_score(resp.content)
                    if score is not None:
                        scores.append(score)
                    else:
                        logger.warning(
                            "LLMJudge[%s]: unparseable response for task=%s: %r",
                            _template_name,
                            task.task_id,
                            resp.content[:120],
                        )
                except Exception as exc:
                    logger.warning(
                        "LLMJudge[%s]: call failed for task=%s: %s",
                        _template_name,
                        task.task_id,
                        exc,
                    )

        if not scores:
            logger.error(
                "LLMJudge: all %d calls failed for task=%s — returning 0.0",
                4 * self._n_samples,
                task.task_id,
            )
            return 0.0

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        if std > self._conf_threshold:
            logger.warning(
                "LLMJudge: LOW CONFIDENCE for task=%s "
                "(mean=%.3f, std=%.3f, n=%d, scores=%s)",
                task.task_id,
                mean,
                std,
                len(scores),
                [round(s, 3) for s in scores],
            )

        return mean
