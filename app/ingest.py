from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from app.chunking import Chunk, chunk_text, discover_documents
from app.config import ConfigError, load_config
from app.logging_setup import configure_logging, correlation_context, get_logger, log_event
from app.providers import GeminiEmbeddingProvider, ProviderError
from app.store import StoreError, VectorStore

logger = get_logger(__name__)


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _citation_path(path: Path, corpus_root: Path) -> str:
    return path.relative_to(corpus_root).as_posix()


def run(config_path: str | None = None, source_dir: str | None = None) -> int:
    started = time.perf_counter()
    config = load_config(config_path)
    configure_logging(config.logging.level)

    source = source_dir or config.corpus.source_dir
    documents, skipped = discover_documents(source, config.corpus.supported_extensions)

    for path in skipped:
        log_event(
            logger,
            "ingest.file_skipped",
            level=logging.WARNING,
            source_path=str(path),
            reason="unsupported extension",
        )

    if not documents:
        log_event(
            logger,
            "ingest.no_documents",
            level=logging.ERROR,
            source_dir=str(source),
        )
        return 1

    all_chunks: list[Chunk] = []
    for path in documents:
        text = path.read_text(encoding="utf-8")
        relative = _citation_path(path, Path(source))
        chunks, warnings = chunk_text(
            text,
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

    if not all_chunks:
        log_event(logger, "ingest.no_chunks", level=logging.ERROR, documents=len(documents))
        return 1

    store = VectorStore(config)
    store.initialise()
    embedder = GeminiEmbeddingProvider(config)

    vectors: list[list[float]] = []
    embedding_calls = 0
    for batch in _batched(all_chunks, config.embedding.embed_batch_size):
        vectors.extend(embedder.embed_documents([c.text for c in batch]))
        embedding_calls += 1

    chunk_count = store.replace_collection(all_chunks, vectors)

    duration_ms = int((time.perf_counter() - started) * 1000)
    log_event(
        logger,
        "ingest.complete",
        documents_processed=len(documents),
        documents_skipped=len(skipped),
        chunks_created=chunk_count,
        embedding_calls=embedding_calls,
        embedding_model=config.embedding.model,
        collection=config.database.collection_name,
        duration_ms=duration_ms,
    )
    return 0


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
            return 2
        except ProviderError as exc:
            log_event(
                logger,
                "ingest.failed",
                level=logging.ERROR,
                stage=exc.stage,
                error_class=type(exc).__name__,
                detail=str(exc),
            )
            return 3
        except StoreError as exc:
            log_event(
                logger,
                "ingest.failed",
                level=logging.ERROR,
                stage="store",
                error_class=type(exc).__name__,
                detail=str(exc),
            )
            return 4
        except FileNotFoundError as exc:
            log_event(logger, "ingest.failed", level=logging.ERROR, stage="read", detail=str(exc))
            return 5


if __name__ == "__main__":
    sys.exit(main())
