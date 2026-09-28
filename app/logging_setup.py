from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "urllib3", "google_genai", "google.genai")
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
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
            payload["error_class"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
            )
        return json.dumps(payload, default=str, ensure_ascii=False)


_OWNED = "_rag_json_handler"


def configure_logging(level: str = "INFO") -> None:
    """Replaces only its own handler, so sinks installed by a host process survive."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _OWNED, True)
    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _OWNED, False)]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    _silence_third_party_loggers()
    _route_uvicorn_through_json_handler()


def _silence_third_party_loggers() -> None:
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _route_uvicorn_through_json_handler() -> None:
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"event": event, **fields})


def new_correlation_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def correlation_context(correlation_id: str | None = None) -> Iterator[str]:
    cid = correlation_id or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class Timer:
    def __init__(self) -> None:
        self.duration_ms: int = 0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.duration_ms = int((time.perf_counter() - self._start) * 1000)
