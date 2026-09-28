from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Config
from app.logging_setup import get_logger, log_event
from app.providers import EmbeddingProvider, GenerationProvider
from app.store import RetrievedChunk

logger = get_logger(__name__)

INSUFFICIENT_CONTEXT_ANSWER = (
    "The documentation does not contain the information needed to answer this question."
)

# Observability only: the defence is the prompt contract, because a pattern list can be evaded.
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above|the)\s+", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"\b(system\s*prompt|developer\s*message)\b", re.I),
    re.compile(r"reply\s+(only\s+)?(with|that)\b", re.I),
    re.compile(r"\bnew\s+instructions?\b", re.I),
)


@dataclass(frozen=True)
class Citation:
    source_path: str
    chunk_ordinal: int
    similarity: float
    snippet: str
    injection_suspected: bool


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    grounded: bool


class ChunkRetriever(Protocol):
    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...


def detect_injection(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(match.group(0).strip())
    return found


def _cite(chunk: RetrievedChunk) -> Citation:
    phrases = detect_injection(chunk.content)
    if phrases:
        log_event(
            logger,
            "prompt_injection_suspected",
            level=logging.WARNING,
            source_path=chunk.source_path,
            chunk_ordinal=chunk.chunk_ordinal,
            matched_phrases=phrases,
        )
    return Citation(
        source_path=chunk.source_path,
        chunk_ordinal=chunk.chunk_ordinal,
        similarity=round(chunk.similarity, 4),
        snippet=chunk.content,
        injection_suspected=bool(phrases),
    )


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f'<document index="{index}" source="{chunk.source_path}" '
            f'chunk="{chunk.chunk_ordinal}">\n{chunk.content}\n</document>'
        )
    return "\n\n".join(parts)


class RagService:
    def __init__(
        self,
        config: Config,
        store: ChunkRetriever,
        embedder: EmbeddingProvider,
        generator: GenerationProvider,
    ) -> None:
        self._config = config
        self._store = store
        self._embedder = embedder
        self._generator = generator
        self._template = self._load_template(config.generation.prompt_template_path)

    @staticmethod
    def _load_template(path: str) -> str:
        template_path = Path(path)
        if not template_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        for placeholder in ("{context}", "{question}"):
            if placeholder not in text:
                raise ValueError(f"Prompt template {template_path} is missing {placeholder}")
        return text

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        k = top_k if top_k is not None else self._config.retrieval.top_k

        query_vector = self._embedder.embed_query(question)
        retrieved = self._store.search(query_vector, k)

        top_score = retrieved[0].similarity if retrieved else 0.0
        threshold = self._config.retrieval.min_similarity
        kept = [chunk for chunk in retrieved if chunk.similarity >= threshold]

        log_event(
            logger,
            "retrieval.result",
            chunks_retrieved=len(retrieved),
            chunks_above_threshold=len(kept),
            top_score=round(top_score, 4),
            below_threshold=not kept,
        )

        if not kept:
            return Answer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                citations=[],
                grounded=False,
            )

        citations = [_cite(chunk) for chunk in kept]
        prompt = self._template.replace("{context}", build_context(kept)).replace(
            "{question}", question
        )
        return Answer(answer=self._generator.generate(prompt), citations=citations, grounded=True)
