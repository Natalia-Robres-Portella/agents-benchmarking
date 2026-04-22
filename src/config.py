"""
Experiment configuration: Pydantic models and YAML loader.

Usage:
    cfg = load_config("configs/experiments/react_hotpotqa.yaml")
    h   = compute_config_hash(cfg)   # deterministic SHA-256
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024


class MemoryConfig(BaseModel):
    type: Literal["no_memory", "window_buffer", "episodic", "vector_store"] = "no_memory"
    window_size: int = 10
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5


class ToolConfig(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    strategy: Literal["direct", "react", "reflexion", "plan_execute", "tot"] = "react"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: List[ToolConfig] = Field(default_factory=list)


class TaskFilterConfig(BaseModel):
    difficulty: Optional[List[str]] = None


class TaskConfig(BaseModel):
    dataset: str = "hotpotqa"
    split: str = "validation"
    n_samples: int = 10
    filter: Optional[TaskFilterConfig] = None


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(
        default_factory=lambda: ["success_rate", "pass_at_k", "tokens_per_task"]
    )
    validator: Literal["exact_match", "fuzzy_match", "functional", "llm_judge"] = "fuzzy_match"
    pass_k_values: List[int] = Field(default_factory=lambda: [1, 3, 5])
    llm_judge_model: Optional[str] = None


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    save_traces: bool = True
    trace_format: Literal["jsonl", "json"] = "jsonl"


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    id: str                                         # required — no default
    seed: int = 42
    n_trials: int = 5
    max_steps: int = 25
    output_dir: str = "results/"
    tags: List[str] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tasks: TaskConfig = Field(default_factory=TaskConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load an experiment config by deep-merging base_config.yaml with the
    given experiment file.  Fields in the experiment file take precedence.
    """
    base_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    merged: Dict[str, Any] = {}

    if base_path.exists():
        with open(base_path) as f:
            merged = yaml.safe_load(f) or {}

    with open(path) as f:
        experiment_data = yaml.safe_load(f) or {}

    _deep_merge(merged, experiment_data)
    # The YAML uses an `experiment:` block for top-level scalars (id, seed, …).
    # Flatten it so ExperimentConfig sees them at the root level.
    if "experiment" in merged:
        _deep_merge(merged, merged.pop("experiment"))
    return ExperimentConfig.model_validate(merged)


def compute_config_hash(config: ExperimentConfig) -> str:
    """Deterministic SHA-256 of the full config.  Embedded in every trajectory."""
    data = config.model_dump()
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge `override` into `base` in-place.  Lists are replaced."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
