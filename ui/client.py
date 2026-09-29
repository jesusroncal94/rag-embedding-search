from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ui.config import UiSettings

_UPSTREAM_MESSAGES = {
    502: "The model provider failed (502).",
    504: "The model provider did not respond in time (504).",
}


@dataclass(frozen=True)
class Citation:
    source_path: str
    chunk_ordinal: int
    similarity: float
    snippet: str
    injection_suspected: bool


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    generated: bool
    correlation_id: str


@dataclass(frozen=True)
class Failure:
    message: str
    correlation_id: str | None


@dataclass(frozen=True)
class Health:
    reachable: bool
    chunk_count: int


class ApiClient:
    def __init__(self, http: httpx.Client, health_timeout_s: float = 3.0) -> None:
        self._http = http
        self._health_timeout_s = health_timeout_s

    def ask(self, question: str) -> Answer | Failure:
        try:
            response = self._http.post("/api/v1/query", json={"question": question})
        except httpx.HTTPError:
            return Failure(f"The API is not reachable at `{self._http.base_url}`.", None)
        body = _json(response)
        if response.status_code == 200:
            return _answer(body)
        return Failure(_failure_message(response.status_code, body), body.get("correlation_id"))

    def health(self) -> Health:
        try:
            body = _json(self._http.get("/health", timeout=self._health_timeout_s))
        except httpx.HTTPError:
            return Health(reachable=False, chunk_count=0)
        database = body.get("vector_database", {})
        return Health(
            reachable=bool(database.get("reachable")),
            chunk_count=int(database.get("chunk_count", 0)),
        )


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _answer(body: dict[str, Any]) -> Answer:
    return Answer(
        text=body["answer"],
        citations=[Citation(**citation) for citation in body["citations"]],
        grounded=body["grounded"],
        generated=body["generated"],
        correlation_id=body["correlation_id"],
    )


def _failure_message(status_code: int, body: dict[str, Any]) -> str:
    if status_code == 400:
        return body.get("detail", "The question was rejected (400).")
    return _UPSTREAM_MESSAGES.get(status_code, f"Unexpected response from the API ({status_code}).")


def create_client() -> ApiClient:
    settings = UiSettings.from_env()
    timeout = httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s)
    return ApiClient(
        httpx.Client(base_url=settings.api_base_url, timeout=timeout),
        health_timeout_s=settings.health_timeout_s,
    )
