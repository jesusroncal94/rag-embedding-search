from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.chunking import Chunk
from app.config import Config
from app.logging_setup import get_logger, log_event

logger = get_logger(__name__)


class StoreError(RuntimeError):
    pass


class EmbeddingModelMismatch(StoreError):
    """Mismatched embeddings raise no error; they return plausible, wrong neighbours."""


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source_path: str
    chunk_ordinal: int
    start_offset: int
    end_offset: int
    similarity: float


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    embedding_model: str
    dimensions: int
    chunk_count: int


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS collections (
    name            TEXT PRIMARY KEY,
    embedding_model TEXT        NOT NULL,
    dimensions      INTEGER     NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    collection    TEXT    NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source_path   TEXT    NOT NULL,
    chunk_ordinal INTEGER NOT NULL,
    start_offset  INTEGER NOT NULL,
    end_offset    INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    embedding     VECTOR({dimensions}) NOT NULL,
    UNIQUE (collection, source_path, chunk_ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection);
"""


class VectorStore:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._url = config.database.url
        self._collection = config.database.collection_name
        self._dimensions = config.embedding.dimensions
        self._model = config.embedding.model

    def _connect(self, *, register_types: bool = True) -> psycopg.Connection:
        """The pgvector adapter cannot register before CREATE EXTENSION has run."""
        try:
            conn = psycopg.connect(self._url, row_factory=dict_row)
        except psycopg.Error as exc:
            raise StoreError(f"vector database unreachable: {type(exc).__name__}") from exc
        if register_types:
            register_vector(conn)
        return conn

    def initialise(self) -> None:
        with self._connect(register_types=False) as conn, conn.cursor() as cur:
            cur.execute(_SCHEMA.format(dimensions=self._dimensions))
            conn.commit()

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None
        except StoreError:
            return False

    def collection_info(self) -> CollectionInfo | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, embedding_model, dimensions FROM collections WHERE name = %s",
                (self._collection,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute("SELECT count(*) AS n FROM chunks WHERE collection = %s", (self._collection,))
            count_row = cur.fetchone()
        return CollectionInfo(
            name=row["name"],
            embedding_model=row["embedding_model"],
            dimensions=row["dimensions"],
            chunk_count=int(count_row["n"]) if count_row else 0,
        )

    def verify_embedding_model(self) -> None:
        info = self.collection_info()
        if info is None:
            log_event(
                logger,
                "store.collection_absent",
                level=logging.WARNING,
                collection=self._collection,
                detail="no corpus ingested yet; queries will return insufficient context",
            )
            return
        if info.embedding_model != self._model or info.dimensions != self._dimensions:
            raise EmbeddingModelMismatch(
                f"Collection '{self._collection}' was built with embedding model "
                f"'{info.embedding_model}' ({info.dimensions}d) but the service is configured "
                f"with '{self._model}' ({self._dimensions}d). Re-run ingestion or restore the "
                f"previous embedding configuration; querying across models returns plausible "
                f"but wrong results."
            )

    def replace_collection(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise StoreError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} vectors"
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO collections (name, embedding_model, dimensions)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE
                        SET embedding_model = EXCLUDED.embedding_model,
                            dimensions      = EXCLUDED.dimensions
                    """,
                    (self._collection, self._model, self._dimensions),
                )
                cur.execute("DELETE FROM chunks WHERE collection = %s", (self._collection,))
                with cur.copy(
                    "COPY chunks (collection, source_path, chunk_ordinal, start_offset, "
                    "end_offset, content, embedding) FROM STDIN WITH (FORMAT BINARY)"
                ) as copy:
                    copy.set_types(
                        ["text", "text", "int4", "int4", "int4", "text", "vector"]
                    )
                    for chunk, vector in zip(chunks, embeddings):
                        copy.write_row(
                            (
                                self._collection,
                                chunk.source_path,
                                chunk.chunk_ordinal,
                                chunk.start_offset,
                                chunk.end_offset,
                                chunk.text,
                                vector,
                            )
                        )
            conn.commit()
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source_path, chunk_ordinal, start_offset, end_offset,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE collection = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, self._collection, query_embedding, top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                content=row["content"],
                source_path=row["source_path"],
                chunk_ordinal=row["chunk_ordinal"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]
