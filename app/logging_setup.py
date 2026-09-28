"""Structured JSON logging — critical events only (FR-LOG-01..07).

Deliberately not a firehose: the spec names six event classes and this module makes
emitting anything else awkward on purpose. Every record is one line of JSON carrying
timestamp, level and event (FR-LOG-07).
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

# Correlation id for the in-flight request (FR-API-10, FR-LOG-02..05).
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One line of JSON per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        cid = _correlation_id.get()
        if cid:
            payload["correlation_id"] = cid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "event":
                payload[key] = value
        if record.exc_info:
            # Class name only — never a stack trace in the log payload (FR-API-08).
            payload["error_class"] = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
        return json.dumps(payload, default=str, ensure_ascii=False)


_OWNED = "_rag_json_handler"


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler, replacing only a previously installed one.

    Deliberately does not clear every root handler: this is called both at service
    start and at the start of an ingestion run, and wiping handlers installed by a
    host process (a test harness, or a supervisor that adds its own sink) would
    silently discard their output.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _OWNED, True)
    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _OWNED, False)]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party libraries are not permitted to pollute the critical-events stream.
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "google.genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # uvicorn installs its own plain-text handlers. Strip them and let its records
    # propagate to the JSON handler above, so FR-LOG-07 holds for every line the
    # container emits - not just the ones this application writes itself.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured record. `event` is the stable machine-readable name."""
    logger.log(level, event, extra={"event": event, **fields})


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def get_correlation_id() -> str | None:
    return _correlation_id.get()


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id to everything logged inside this block."""
    cid = correlation_id or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class Timer:
    """Wall-clock timer for provider latency logging (FR-LOG-03/04)."""

    def __init__(self) -> None:
        self.duration_ms: int = 0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.duration_ms = int((time.perf_counter() - self._start) * 1000)
