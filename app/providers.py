from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeVar

from app.config import Config
from app.logging_setup import Timer, get_logger, log_event

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

Stage = Literal["embed", "generate"]
T = TypeVar("T")


class ProviderError(RuntimeError):
    def __init__(self, stage: Stage, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class ProviderTimeout(ProviderError):
    pass


class ProviderAuthError(ProviderError):
    pass


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
    if isinstance(exc, TimeoutError | ProviderTimeout):
        return True
    return "timeout" in type(exc).__name__.lower() or "deadline" in str(exc).lower()


def _retryable(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    return _is_timeout(exc)


def _rpc_details(exc: Exception) -> list[dict]:
    body = getattr(exc, "details", None)
    if not isinstance(body, dict):
        return []
    details = body.get("error", body).get("details", [])
    return [item for item in details if isinstance(item, dict)]


def _daily_quota_exhausted(exc: Exception) -> bool:
    return any(
        "PerDay" in str(violation.get("quotaId", ""))
        for item in _rpc_details(exc)
        for violation in item.get("violations", [])
    )


def _retry_after_ms(exc: Exception) -> int | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    header = headers.get("Retry-After")
    if header and header.strip().isdigit():
        return int(header) * 1000
    for item in _rpc_details(exc):
        delay = str(item.get("retryDelay", ""))
        if delay.endswith("s"):
            try:
                return int(float(delay[:-1]) * 1000)
            except ValueError:
                return None
    return None


def with_retries(
    operation: Callable[[], T],
    *,
    stage: Stage,
    max_retries: int,
    backoff_base_ms: int,
    max_wait_ms: int,
) -> T:
    """A 429 asks for less traffic: retrying it early spends the quota that is limiting us."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            status = _status_of(exc)
            if status in (401, 403):
                raise ProviderAuthError(stage, f"credentials rejected (HTTP {status})") from exc
            if _daily_quota_exhausted(exc):
                raise ProviderError(stage, "daily quota exhausted (HTTP 429)") from exc
            retry_after_ms = _retry_after_ms(exc)
            if retry_after_ms is not None and retry_after_ms > max_wait_ms:
                raise ProviderError(
                    stage, f"rate limited for {retry_after_ms} ms, longer than the wait allowed"
                ) from exc
            if not _retryable(exc) or attempt == max_retries:
                raise _as_provider_error(stage, exc) from exc
            delay_ms = retry_after_ms or _backoff_ms(attempt, backoff_base_ms)
            _wait_before_retry(stage, attempt, delay_ms, exc)
            attempt += 1


def _as_provider_error(stage: Stage, exc: Exception) -> ProviderError:
    if _is_timeout(exc):
        return ProviderTimeout(stage, "upstream timed out")
    return ProviderError(stage, f"upstream call failed ({type(exc).__name__})")


def _backoff_ms(attempt: int, backoff_base_ms: int) -> float:
    return backoff_base_ms * (2 ** (attempt - 1)) + random.uniform(0, backoff_base_ms * 0.1)


def _wait_before_retry(stage: Stage, attempt: int, delay_ms: float, exc: Exception) -> None:
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


def _finish_reason(response: object) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return getattr(reason, "name", reason)


def _gemini_client(config: Config) -> Any:
    from google import genai

    return genai.Client(api_key=config.gemini_api_key)


class GeminiEmbeddingProvider:
    """Queries and documents use different task types: retrieval embeddings are asymmetric."""

    def __init__(self, config: Config, client: Any | None = None) -> None:
        from google.genai import types

        self._types = types
        self._config = config
        self._client = client or _gemini_client(config)
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
                max_wait_ms=self._config.runtime.request_timeout_ms,
            )
        log_event(
            logger,
            "embedding.latency",
            duration_ms=timer.duration_ms,
            batch_size=len(texts),
            model=self._model,
        )
        if len(vectors) != len(texts):
            raise ProviderError(
                "embed", f"provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self._config.embedding.document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._config.embedding.query_task_type)[0]


class GeminiGenerationProvider:
    def __init__(self, config: Config, client: Any | None = None) -> None:
        from google.genai import types

        self._types = types
        self._config = config
        self._client = client or _gemini_client(config)
        self._model = config.generation.model
        # No tools are passed, yet the SDK runs function calling by default and logs it to
        # the root logger on every call. maximum_remote_calls must be None, or disabling
        # it logs a warning instead.
        self._no_function_calling = types.AutomaticFunctionCallingConfig(
            disable=True, maximum_remote_calls=None
        )

    def generate(self, prompt: str) -> str:
        def call() -> tuple[str, object]:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self._config.generation.temperature,
                    max_output_tokens=self._config.generation.max_output_tokens,
                    http_options=self._types.HttpOptions(
                        timeout=self._config.runtime.request_timeout_ms
                    ),
                    automatic_function_calling=self._no_function_calling,
                ),
            )
            return (response.text or "").strip(), response

        with Timer() as timer:
            text, response = with_retries(
                call,
                stage="generate",
                max_retries=self._config.runtime.max_retries,
                backoff_base_ms=self._config.runtime.backoff_base_ms,
                max_wait_ms=self._config.runtime.request_timeout_ms,
            )

        usage = getattr(response, "usage_metadata", None)
        finish_reason = _finish_reason(response)
        log_event(
            logger,
            "llm.latency",
            duration_ms=timer.duration_ms,
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            finish_reason=finish_reason,
        )
        if finish_reason == "MAX_TOKENS":
            raise ProviderError("generate", "answer truncated at max_output_tokens")
        if not text:
            raise ProviderError("generate", "provider returned an empty response")
        return text
