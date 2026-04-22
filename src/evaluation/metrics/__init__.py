from src.evaluation.metrics.base import METRIC_REGISTRY, Metric, MetricRegistry
from src.evaluation.metrics.failure_recovery import FailureRecoveryMetric
from src.evaluation.metrics.latency import LatencyMetric
from src.evaluation.metrics.pass_at_k import PassAtKMetric
from src.evaluation.metrics.steps import StepCountMetric
from src.evaluation.metrics.success_rate import SuccessRateMetric
from src.evaluation.metrics.tokens import TokensPerTaskMetric
from src.evaluation.metrics.tool_accuracy import ToolAccuracyMetric

METRIC_REGISTRY.register(SuccessRateMetric)
METRIC_REGISTRY.register(PassAtKMetric)
METRIC_REGISTRY.register(TokensPerTaskMetric)
METRIC_REGISTRY.register(StepCountMetric)
METRIC_REGISTRY.register(ToolAccuracyMetric)
METRIC_REGISTRY.register(FailureRecoveryMetric)
METRIC_REGISTRY.register(LatencyMetric)

__all__ = [
    "Metric",
    "MetricRegistry",
    "METRIC_REGISTRY",
    "SuccessRateMetric",
    "PassAtKMetric",
    "TokensPerTaskMetric",
    "StepCountMetric",
    "ToolAccuracyMetric",
    "FailureRecoveryMetric",
    "LatencyMetric",
]
