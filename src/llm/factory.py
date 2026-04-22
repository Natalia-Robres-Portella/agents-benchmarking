"""Build an LLMBackend from an LLMConfig."""
from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMBackend


def build_llm_backend(config: LLMConfig) -> LLMBackend:
    if config.provider == "openai":
        from src.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(config)
    if config.provider == "anthropic":
        from src.llm.anthropic_backend import AnthropicBackend  # advanced
        return AnthropicBackend(config)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
