"""Shared fixtures and test doubles.

Execution tiers (ADR-003), selected by marker:
  @unit    - runs anywhere, no network, no Docker. The default suite.
  @live    - asserts real model behaviour; needs GEMINI_API_KEY and an ingested corpus.
  @docker  - asserts compose/runtime behaviour; needs a Docker daemon.
  @manual  - human judgement (NFR-05 readability); never automated.

`pytest` runs @unit only. The others are opt-in via `-m live` / `-m docker`, which is
deliberate: a suite that silently skips is honest only if the skip is visible, so the
tiers are named in the spec and in the README rather than hidden behind a conftest flag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pytest
import yaml

from app.config import Config, load_config
from app.logging_setup import configure_logging
from app.providers import ProviderAuthError, ProviderError, ProviderTimeout
from app.store import CollectionInfo, RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "docs" / "spec" / "features"
CORPUS_DIR = REPO_ROOT / "corpus"

_DIMENSIONS = 64


def pytest_configure(config: pytest.Config) -> None:
    for marker, description in (
        ("unit", "no network or Docker required"),
        ("live", "requires GEMINI_API_KEY and a real model"),
        ("docker", "requires a running Docker daemon"),
        ("manual", "human judgement; not automated"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = config.getoption("-m")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="manual scenario - see SPEC.md NFR-05"))
        if "live" in item.keywords and not selected and not os.environ.get("GEMINI_API_KEY"):
            item.add_marker(pytest.mark.skip(reason="live tier: set GEMINI_API_KEY and run -m live"))
        if "docker" in item.keywords and not selected:
            item.add_marker(pytest.mark.skip(reason="docker tier: run with -m docker"))


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


def _fake_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding: a hashed bag of words, L2-normalised.

    Lexical overlap raises cosine similarity, which is enough to exercise retrieval
    plumbing, thresholds and ordering. It is NOT semantic — which is exactly why the
    scenarios asserting semantic behaviour are tagged @live rather than faked here.
    """
    vector = [0.0] * _DIMENSIONS
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "big") % _DIMENSIONS] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * _DIMENSIONS


class FakeEmbedder:
    """Records calls so batching (FR-ING-07) and task types can be asserted.

    Wraps its work in the same `with_retries` helper the real provider uses, so that
    retry and backoff behaviour (FR-ING-08) is exercised against the production policy
    rather than a reimplementation of it.
    """

    def __init__(self, *, failures: list[Exception] | None = None, max_retries: int = 3) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self._failures = list(failures or [])
        self._max_retries = max_retries

    def _maybe_fail(self) -> None:
        if self._failures:
            raise self._failures.pop(0)

    def _guarded(self, operation):
        from app.providers import with_retries

        return with_retries(
            operation, stage="embed", max_retries=self._max_retries, backoff_base_ms=1
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            self._maybe_fail()
            self.document_calls.append(list(texts))
            return [_fake_vector(text) for text in texts]

        return self._guarded(call)

    def embed_query(self, text: str) -> list[float]:
        def call() -> list[float]:
            self._maybe_fail()
            self.query_calls.append(text)
            return _fake_vector(text)

        return self._guarded(call)

    @property
    def call_count(self) -> int:
        return len(self.document_calls)


class FakeGenerator:
    """Echoes a canned answer and captures the prompt for inspection."""

    def __init__(self, *, answer: str = "A grounded answer.", error: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self._answer = answer
        self._error = error

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        return self._answer

    @property
    def call_count(self) -> int:
        return len(self.prompts)


class InMemoryStore:
    """VectorStore-compatible double backed by a list.

    Mirrors the real store's contract exactly — including cosine similarity and the
    collection-metadata guard — so that step definitions bind to behaviour rather than
    to Postgres.
    """

    def __init__(self, *, embedding_model: str = "fake-embed", dimensions: int = _DIMENSIONS) -> None:
        self.rows: list[tuple] = []
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.collection_name = "docs"
        self.reachable = True
        self.has_collection = False
        self.replace_calls = 0

    def initialise(self) -> None:
        return None

    def ping(self) -> bool:
        return self.reachable

    def collection_info(self) -> CollectionInfo | None:
        if not self.has_collection:
            return None
        return CollectionInfo(
            name=self.collection_name,
            embedding_model=self.embedding_model,
            dimensions=self.dimensions,
            chunk_count=len(self.rows),
        )

    def verify_embedding_model(self) -> None:
        return None

    def replace_collection(self, chunks, embeddings) -> int:
        if len(chunks) != len(embeddings):
            raise AssertionError("chunk/embedding count mismatch")
        self.replace_calls += 1
        self.has_collection = True
        self.rows = [(chunk, vector) for chunk, vector in zip(chunks, embeddings)]
        return len(self.rows)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored = []
        for chunk, vector in self.rows:
            similarity = sum(a * b for a, b in zip(query_embedding, vector))
            scored.append((similarity, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                content=chunk.text,
                source_path=chunk.source_path,
                chunk_ordinal=chunk.chunk_ordinal,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                similarity=float(similarity),
            )
            for similarity, chunk in scored[:top_k]
        ]


class LogCapture:
    """Captures emitted records as parsed JSON, exactly as they hit stdout."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    @property
    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]

    def events(self, name: str) -> list[dict]:
        return [record for record in self.records if record.get("event") == name]

    def contains_text(self, needle: str) -> bool:
        return any(needle in line for line in self.lines)


class _CaptureHandler(logging.Handler):
    def __init__(self, capture: LogCapture, formatter: logging.Formatter) -> None:
        super().__init__()
        self._capture = capture
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        # Only the application's own records. Third-party libraries at DEBUG (asyncio's
        # selector, pytest's fixture resolution) are not part of the service's log
        # contract and would otherwise pollute FR-LOG assertions.
        if not record.name.startswith("app"):
            return
        self._capture.lines.append(self.format(record))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def log_capture() -> LogCapture:
    from app.logging_setup import JsonFormatter

    configure_logging("INFO")
    logging.getLogger("app").setLevel(logging.DEBUG)
    capture = LogCapture()
    handler = _CaptureHandler(capture, JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield capture
    finally:
        root.removeHandler(handler)


@pytest.fixture
def base_config(monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.chdir(REPO_ROOT)
    return load_config(REPO_ROOT / "config" / "config.yaml")


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a config.yaml with overrides applied, returning its path.

    Overrides use dotted keys ("retrieval.top_k"), matching how the feature files
    refer to configuration.
    """

    def _write(overrides: dict[str, object] | None = None) -> Path:
        with (REPO_ROOT / "config" / "config.yaml").open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for dotted, value in (overrides or {}).items():
            section, _, key = dotted.partition(".")
            data[section][key] = value
        # Keep the prompt template resolvable from the temp location.
        data["generation"]["prompt_template_path"] = str(REPO_ROOT / "prompts" / "answer.md")
        target = tmp_path / "config.yaml"
        target.write_text(yaml.safe_dump(data), encoding="utf-8")
        return target

    return _write


@dataclass
class Harness:
    """Everything a scenario needs to drive the service with doubles in place."""

    config: Config
    store: InMemoryStore
    embedder: FakeEmbedder
    generator: FakeGenerator
    logs: LogCapture
    response: object = None
    status_code: int | None = None
    body: dict | None = None
    exit_code: int | None = None
    error: Exception | None = None
    chunks: list = None
    warnings: list = None


@pytest.fixture
def harness(base_config: Config, log_capture: LogCapture) -> Harness:
    return Harness(
        config=replace(base_config),
        store=InMemoryStore(),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
        logs=log_capture,
        chunks=[],
        warnings=[],
    )


@pytest.fixture
def corpus_dir() -> Path:
    if not CORPUS_DIR.is_dir():
        pytest.skip("corpus/ not present")
    return CORPUS_DIR


def make_error(kind: str) -> Exception:
    return {
        "auth": ProviderAuthError("embed: credentials rejected (HTTP 401)"),
        "timeout": ProviderTimeout("generate: upstream timed out"),
        "provider": ProviderError("generate: upstream call failed"),
    }[kind]
