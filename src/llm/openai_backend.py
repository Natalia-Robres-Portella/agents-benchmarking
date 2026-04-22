"""OpenAI LLM backend — wraps openai.chat.completions with retries and token tracking."""
from __future__ import annotations

import os
import time
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import LLMConfig
from src.llm.base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    """
    Single call-point for all OpenAI traffic.
    Retries on rate-limit errors (up to 3 attempts, exponential backoff).
    Token counts are extracted from the usage object and normalised into
    LLMResponse so the execution engine can track cost uniformly.
    """

    def __init__(self, config: LLMConfig) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._config = config
        self._client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "NO_KEY_SET")
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: object,
    ) -> LLMResponse:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stop=stop or None,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            model=resp.model,
            latency_ms=latency_ms,
        )

    @property
    def model_id(self) -> str:
        return self._config.model
