from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from app import api as api_module
from app.chunking import chunk_text
from app.config import Config, load_config
from app.logging_setup import JsonFormatter, configure_logging
from app.providers import EmbeddingProvider, GenerationProvider, with_retries
from app.rag import RagService
from app.store import CollectionInfo, RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = REPO_ROOT / "docs" / "spec" / "features"
CORPUS_DIR = REPO_ROOT / "corpus"

_DIMENSIONS = 64

# The lexical double scores token overlap, not meaning: "What is the capital of Peru?"
# outranks real questions. So the unit tier lowers the floor to exercise the threshold,
# and marks out-of-corpus questions instead of inferring them. It proves the refusal
# mechanism, never the judgement of relevance, which only the @live tier can assert.
LEXICAL_FLOOR = 0.05
REFUSING_FLOOR = 0.99
OUT_OF_CORPUS = frozenset(
    {
        "What is the capital of Peru?",
        "What is the daily ATM withdrawal limit in Japan?",
        "What is the CEO's home address?",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = config.getoption("-m")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="manual scenario - see SPEC.md NFR-05"))
        if "live" in item.keywords and not selected and not os.environ.get("GEMINI_API_KEY"):
            item.add_marker(
                pytest.mark.skip(reason="live tier: set GEMINI_API_KEY and run -m live")
            )
        if "docker" in item.keywords and not selected:
            item.add_marker(pytest.mark.skip(reason="docker tier: run with -m docker"))


def _fake_vector(text: str) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "big") % _DIMENSIONS] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else [0.0] * _DIMENSIONS


class FakeEmbedder:
    """Runs through the production with_retries, so retry behaviour is the real policy."""

    def __init__(self, *, failures: list[Exception] | None = None, max_retries: int = 3) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self._failures = list(failures or [])
        self._max_retries = max_retries
        self.attempts = 0

    def _maybe_fail(self) -> None:
        self.attempts += 1
        if self._failures:
            raise self._failures.pop(0)

    def _guarded(self, operation):
        return with_retries(
            operation,
            stage="embed",
            max_retries=self._max_retries,
            backoff_base_ms=1,
            max_wait_ms=1_000,
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
    def __init__(
        self, *, answer: str = "A grounded answer.", error: Exception | None = None
    ) -> None:
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


class _FakeGenaiModels:
    def __init__(self, answer: str, finish_reason: str) -> None:
        self._answer = answer
        self._finish_reason = finish_reason

    def embed_content(self, *, model: str, contents: list[str], config: object) -> SimpleNamespace:
        self.last_embed_config = config
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=_fake_vector(text)) for text in contents]
        )

    def generate_content(self, *, model: str, contents: str, config: object) -> SimpleNamespace:
        self.last_config = config
        return SimpleNamespace(
            text=self._answer,
            candidates=[SimpleNamespace(finish_reason=self._finish_reason)],
            usage_metadata=SimpleNamespace(
                prompt_token_count=len(contents.split()),
                candidates_token_count=len(self._answer.split()),
                total_token_count=len(contents.split()) + len(self._answer.split()),
            ),
        )


class FakeGenaiClient:
    """Stands in for google.genai.Client, so the real Gemini adapters run end to end."""

    def __init__(self, *, answer: str = "A grounded answer.", finish_reason: str = "STOP") -> None:
        self.models = _FakeGenaiModels(answer, finish_reason)


class InMemoryStore:
    def __init__(
        self, *, embedding_model: str = "fake-embed", dimensions: int = _DIMENSIONS
    ) -> None:
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

    def replace_collection(self, chunks, embeddings) -> int:
        if len(chunks) != len(embeddings):
            raise AssertionError("chunk/embedding count mismatch")
        self.replace_calls += 1
        self.has_collection = True
        self.rows = [(chunk, vector) for chunk, vector in zip(chunks, embeddings, strict=True)]
        return len(self.rows)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored = []
        for chunk, vector in self.rows:
            similarity = sum(a * b for a, b in zip(query_embedding, vector, strict=True))
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
    def __init__(self) -> None:
        self.lines: list[str] = []

    @property
    def records(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]

    def events(self, name: str) -> list[dict]:
        return [record for record in self.records if record.get("event") == name]

    def contains_text(self, needle: str) -> bool:
        return any(needle in line for line in self.lines)


class _AppRecordsHandler(logging.Handler):
    def __init__(self, capture: LogCapture, formatter: logging.Formatter) -> None:
        super().__init__()
        self._capture = capture
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("app"):
            self._capture.lines.append(self.format(record))


def with_retrieval(config: Config, **changes) -> Config:
    return replace(config, retrieval=replace(config.retrieval, **changes))


def seed_corpus(store: InMemoryStore, embedder: EmbeddingProvider, config: Config) -> None:
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        produced, _ = chunk_text(
            path.read_text(encoding="utf-8"),
            source_path=path.name,
            min_chunk_chars=config.chunking.min_chunk_chars,
            max_chunk_chars=config.chunking.max_chunk_chars,
            paragraph_delimiter=config.chunking.paragraph_delimiter,
        )
        chunks.extend(produced)
    store.replace_collection(chunks, embedder.embed_documents([c.text for c in chunks]))


def wire_app(
    config: Config, store: InMemoryStore, embedder: EmbeddingProvider, generator: GenerationProvider
) -> TestClient:
    api_module.app.state.services = api_module.Services(
        config=config,
        index=store,
        rag=RagService(config=config, store=store, embedder=embedder, generator=generator),
    )
    return TestClient(api_module.app)


def wire_ctx(ctx: dict) -> TestClient:
    return wire_app(ctx["config"], ctx["store"], ctx["embedder"], ctx["generator"])


@pytest.fixture
def log_capture() -> LogCapture:
    configure_logging("INFO")
    logging.getLogger("app").setLevel(logging.DEBUG)
    capture = LogCapture()
    handler = _AppRecordsHandler(capture, JsonFormatter())
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
def write_config(tmp_path: Path):
    def _write(overrides: dict[str, object] | None = None) -> Path:
        with (REPO_ROOT / "config" / "config.yaml").open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for dotted, value in (overrides or {}).items():
            section, _, key = dotted.partition(".")
            data[section][key] = value
        data["generation"]["prompt_template_path"] = str(REPO_ROOT / "prompts" / "answer.md")
        target = tmp_path / "config.yaml"
        target.write_text(yaml.safe_dump(data), encoding="utf-8")
        return target

    return _write


@pytest.fixture
def corpus_dir() -> Path:
    if not CORPUS_DIR.is_dir():
        pytest.skip("corpus/ not present")
    return CORPUS_DIR
