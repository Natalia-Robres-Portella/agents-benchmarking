from src.evaluation.validators.base import VALIDATOR_REGISTRY, ValidatorRegistry
from src.evaluation.validators.exact_match import ExactMatchValidator
from src.evaluation.validators.fuzzy_match import FuzzyMatchValidator

# LLMJudgeValidator is NOT registered here — it requires an LLM backend
# injected at construction time.  EvaluationModule handles it specially.
# Import it so it's accessible via this package.
from src.evaluation.validators.llm_judge import LLMJudgeValidator  # noqa: F401

VALIDATOR_REGISTRY.register(ExactMatchValidator)
VALIDATOR_REGISTRY.register(FuzzyMatchValidator)

__all__ = [
    "ValidatorRegistry",
    "VALIDATOR_REGISTRY",
    "ExactMatchValidator",
    "FuzzyMatchValidator",
    "LLMJudgeValidator",
]
