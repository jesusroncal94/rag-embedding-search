from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from app.logging_setup import get_logger, log_event
from app.providers import ProviderError, Stage

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitOpen(ProviderError):
    pass


class CircuitBreaker:
    """In-process, so each api replica keeps its own count."""

    def __init__(
        self,
        stage: Stage,
        *,
        failure_threshold: int,
        open_ms: int,
    ) -> None:
        self._stage = stage
        self._failure_threshold = failure_threshold
        self._open_s = open_ms / 1000.0
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    def call(self, operation: Callable[[], T]) -> T:
        self._admit()
        try:
            result = operation()
        except ProviderError:
            self._record_failure()
            raise
        self._record_success()
        return result

    def _trial_due(self) -> bool:
        return self._opened_at is not None and time.monotonic() - self._opened_at >= self._open_s

    def _admit(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if not self._trial_due() or self._trial_in_flight:
                raise CircuitOpen(self._stage, "circuit open; the provider is not being called")
            self._trial_in_flight = True

    def _record_failure(self) -> None:
        with self._lock:
            self._trial_in_flight = False
            self._failures += 1
            if self._opened_at is not None or self._failures >= self._failure_threshold:
                if self._opened_at is None:
                    log_event(
                        logger,
                        "provider.circuit_opened",
                        level=logging.WARNING,
                        stage=self._stage,
                        consecutive_failures=self._failures,
                    )
                self._opened_at = time.monotonic()

    def _record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                log_event(logger, "provider.circuit_closed", stage=self._stage)
            self._failures = 0
            self._opened_at = None
            self._trial_in_flight = False
