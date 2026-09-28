"""Gemini provider adapters — embedding and generation.

Each sits behind a narrow interface (SPEC.md §5) so it can be swapped or faked. Retry
with exponential backoff on transient failures (FR-ING-08); timeouts are surfaced
distinctly from other upstream failures (FR-API-08a).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Protocol, Sequence

from app.config import Config
from app.logging_setup import Timer, get_logger, log_event

logger = get_logger(__name__)

# HTTP statuses worth retrying: rate limits and transient server faults.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ProviderTimeout(RuntimeError):
    """An upstream call exceeded runtime.request_timeout_ms (FR-API-08a -> HTTP 504)."""


class ProviderError(RuntimeError):
    """An upstream call failed permanently (FR-API-08 -> HTTP 502)."""


class ProviderAuthError(ProviderError):
    """Credentials rejected. Never retried — retrying a 401 only wastes time."""


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class GenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        return response.status_code
    return None


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ProviderTimeout)):
        return True
    return "timeout" in type(exc).__name__.lower() or "deadline" in str(exc).lower()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, ProviderAuthError):
        return False
    status = _status_of(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    return _is_timeout(exc)


def with_retries(operation, *, stage: str, max_retries: int, backoff_base_ms: int):
    """Call `operation`, retrying transient failures with exponential backoff + jitter.

    FR-ING-08. A permanent failure (auth, bad request) is raised on the first attempt;
    retrying it would only slow the failure down.
    """
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - normalised into provider errors below
            last = exc
            status = _status_of(exc)
            if status == 401 or status == 403:
                raise ProviderAuthError(f"{stage}: credentials rejected (HTTP {status})") from exc
            if not _retryable(exc) or attempt == max_retries:
                break
            delay_ms = backoff_base_ms * (2 ** (attempt - 1))
            delay_ms += random.uniform(0, backoff_base_ms * 0.1)
            log_event(
                logger,
                "provider.retry",
                level=logging.WARNING,
                stage=stage,
                attempt=attempt,
                backoff_ms=int(delay_ms),
                error_class=type(exc).__name__,
            )
            time.sleep(delay_ms / 1000.0)

    assert last is not None
    if _is_timeout(last):
        raise ProviderTimeout(f"{stage}: upstream timed out") from last
    raise ProviderError(f"{stage}: upstream call failed ({type(last).__name__})") from last


class GeminiEmbeddingProvider:
    """Gemini Embedding (FR-ING-05, FR-API-02).

    Documents and queries are embedded with different task types — Gemini's retrieval
    embeddings are asymmetric, and using RETRIEVAL_DOCUMENT for a query measurably
    degrades recall. This is a quiet source of "retrieval just isn't very good" in RAG
    systems, so it is made explicit here.
    """

    def __init__(self, config: Config) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.embedding.model
        self._dimensions = config.embedding.dimensions

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        def call() -> list[list[float]]:
            response = self._client.models.embed_content(
                model=self._model,
                contents=list(texts),
                config=self._types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                ),
            )
            return [list(item.values) for item in response.embeddings]

        with Timer() as timer:
            vectors = with_retries(
                call,
                stage="embed",
                max_retries=self._config.runtime.max_retries,
                backoff_base_ms=self._config.runtime.backoff_base_ms,
            )
        # FR-LOG-04
        log_event(
            logger,
            "embedding.latency",
            duration_ms=timer.duration_ms,
            batch_size=len(texts),
            model=self._model,
        )
        if len(vectors) != len(texts):
            raise ProviderError(
                f"embed: provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self._config.embedding.document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._config.embedding.query_task_type)[0]


class GeminiGenerationProvider:
    """Gemini 2.5 Flash (FR-API-04)."""

    def __init__(self, config: Config) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.generation.model

    def generate(self, prompt: str) -> str:
        def call() -> str:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self._config.generation.temperature,
                    max_output_tokens=self._config.generation.max_output_tokens,
                    http_options=self._types.HttpOptions(
                        timeout=self._config.runtime.request_timeout_ms
                    ),
                ),
            )
            return (response.text or "").strip(), response

        with Timer() as timer:
            text, response = with_retries(
                call,
                stage="generate",
                max_retries=self._config.runtime.max_retries,
                backoff_base_ms=self._config.runtime.backoff_base_ms,
            )

        usage = getattr(response, "usage_metadata", None)
        # FR-LOG-03
        log_event(
            logger,
            "llm.latency",
            duration_ms=timer.duration_ms,
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
        if not text:
            raise ProviderError("generate: provider returned an empty response")
        return text
