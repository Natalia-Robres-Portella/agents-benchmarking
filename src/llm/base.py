"""LLM backend contract.  All providers implement LLMBackend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Normalised response from any LLM provider."""
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLMBackend(ABC):
    """
    Single chokepoint for all LLM traffic.
    Every call — retries, cost tracking, caching — happens here.
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: object,
    ) -> LLMResponse:
        """Send `prompt` to the model; return a normalised response."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier (e.g. 'gpt-4o', 'claude-opus-4-7')."""
        ...
