from __future__ import annotations

import argparse
import logging
import sys
import time
from enum import IntEnum
from pathlib import Path

from app.chunking import Chunk, chunk_text, discover_documents
from app.config import Config, ConfigError, load_config
from app.logging_setup import configure_logging, correlation_context, get_logger, log_event
from app.providers import EmbeddingProvider, GeminiEmbeddingProvider, ProviderError
from app.store import StoreError, VectorStore

logger = get_logger(__name__)


class ExitCode(IntEnum):
    OK = 0
    NOTHING_TO_INGEST = 1
    INVALID_CONFIG = 2
    PROVIDER_FAILED = 3
    STORE_FAILED = 4
    SOURCE_MISSING = 5


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _citation_path(path: Path, corpus_root: Path) -> str:
    return path.relative_to(corpus_root).as_posix()


def _log_skipped(paths: list[Path]) -> None:
    for path in paths:
        log_event(
            logger,
            "ingest.file_skipped",
            level=logging.WARNING,
            source_path=str(path),
            reason="unsupported extension",
        )


def _chunk_documents(documents: list[Path], source: Path, config: Config) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in documents:
        relative = _citation_path(path, source)
        chunks, warnings = chunk_text(
            path.read_text(encoding="utf-8"),
            source_path=relative,
            min_chunk_chars=config.chunking.min_chunk_chars,
            max_chunk_chars=config.chunking.max_chunk_chars,
            paragraph_delimiter=config.chunking.paragraph_delimiter,
        )
        for warning in warnings:
            log_event(logger, "ingest.hard_split", level=logging.WARNING, detail=warning)
        log_event(
            logger,
            "ingest.document_chunked",
            level=logging.DEBUG,
            source_path=relative,
            chunks=len(chunks),
        )
        all_chunks.extend(chunks)
    return all_chunks


def _embed(
    chunks: list[Chunk], embedder: EmbeddingProvider, batch_size: int
) -> tuple[list[list[float]], int]:
    vectors: list[list[float]] = []
    calls = 0
    for batch in _batched(chunks, batch_size):
        vectors.extend(embedder.embed_documents([c.text for c in batch]))
        calls += 1
    return vectors, calls


def run(config_path: str | None = None, source_dir: str | None = None) -> ExitCode:
    started = time.perf_counter()
    config = load_config(config_path)
    configure_logging(config.logging.level)

    source = Path(source_dir or config.corpus.source_dir)
    documents, skipped = discover_documents(source, config.corpus.supported_extensions)
    _log_skipped(skipped)
    if not documents:
        log_event(logger, "ingest.no_documents", level=logging.ERROR, source_dir=str(source))
        return ExitCode.NOTHING_TO_INGEST

    chunks = _chunk_documents(documents, source, config)
    if not chunks:
        log_event(logger, "ingest.no_chunks", level=logging.ERROR, documents=len(documents))
        return ExitCode.NOTHING_TO_INGEST

    store = VectorStore(config)
    store.initialise()
    vectors, embedding_calls = _embed(
        chunks, GeminiEmbeddingProvider(config), config.embedding.embed_batch_size
    )
    chunk_count = store.replace_collection(chunks, vectors)

    log_event(
        logger,
        "ingest.complete",
        documents_processed=len(documents),
        documents_skipped=len(skipped),
        chunks_created=chunk_count,
        embedding_calls=embedding_calls,
        embedding_model=config.embedding.model,
        collection=config.database.collection_name,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return ExitCode.OK


def _failed(stage: str, exc: Exception, code: ExitCode) -> ExitCode:
    log_event(
        logger,
        "ingest.failed",
        level=logging.ERROR,
        stage=stage,
        error_class=type(exc).__name__,
        detail=str(exc),
    )
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--source", default=None, help="Override corpus.source_dir")
    args = parser.parse_args(argv)

    with correlation_context():
        try:
            return run(args.config, args.source)
        except ConfigError as exc:
            configure_logging()
            log_event(logger, "ingest.config_error", level=logging.ERROR, detail=str(exc))
            return ExitCode.INVALID_CONFIG
        except ProviderError as exc:
            return _failed(exc.stage, exc, ExitCode.PROVIDER_FAILED)
        except StoreError as exc:
            return _failed("store", exc, ExitCode.STORE_FAILED)
        except FileNotFoundError as exc:
            return _failed("read", exc, ExitCode.SOURCE_MISSING)


if __name__ == "__main__":
    sys.exit(main())
