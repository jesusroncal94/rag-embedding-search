"""The RAG flow: embed -> retrieve -> fence -> generate (FR-API-02..06a, 11..13)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import Config
from app.logging_setup import get_logger, log_event
from app.providers import EmbeddingProvider, GenerationProvider
from app.store import RetrievedChunk, VectorStore

logger = get_logger(__name__)

INSUFFICIENT_CONTEXT_ANSWER = (
    "The documentation does not contain the information needed to answer this question."
)

# Heuristics for instruction-like text inside retrieved documents (FR-API-12).
# Detection is for OBSERVABILITY, not defence: the actual mitigation is the prompt
# contract in prompts/answer.md, which treats the whole context block as data. A
# detector is a filter that can be evaded; a model instructed never to obey retrieved
# text does not depend on spotting the payload. We log so that a poisoned document can
# be found and removed at source.
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


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    grounded: bool


def detect_injection(text: str) -> list[str]:
    """Return the instruction-like phrases found in `text`."""
    found: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append(match.group(0).strip())
    return found


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Fence each chunk as attributed data (FR-API-11).

    Every chunk is wrapped in its own tagged element carrying its source. This gives the
    model a structural signal for "this is quoted material" and supplies the exact
    identifier the citation rule in the prompt asks for.
    """
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
        store: VectorStore,
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

        # FR-LOG-05
        log_event(
            logger,
            "retrieval.result",
            chunks_retrieved=len(retrieved),
            chunks_above_threshold=len(kept),
            top_score=round(top_score, 4),
            below_threshold=not kept,
        )

        # FR-API-06/06a: deterministic refusal. No LLM call at all — it costs nothing,
        # cannot hallucinate, and cannot be steered by a question crafted to retrieve
        # nothing.
        if not kept:
            return Answer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                citations=[],
                grounded=False,
            )

        context = build_context(kept)

        # FR-API-12: observability on poisoned source documents.
        for chunk in kept:
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

        prompt = self._template.replace("{context}", context).replace("{question}", question)
        answer_text = self._generator.generate(prompt)

        return Answer(
            answer=answer_text,
            citations=[
                Citation(
                    source_path=chunk.source_path,
                    chunk_ordinal=chunk.chunk_ordinal,
                    similarity=round(chunk.similarity, 4),
                )
                for chunk in kept
            ],
            grounded=True,
        )
