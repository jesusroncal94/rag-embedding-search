from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator, Protocol

from fastapi import Depends, FastAPI, Header, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import __version__
from app.config import Config, load_config
from app.logging_setup import (
    configure_logging,
    correlation_context,
    get_logger,
    log_event,
    new_correlation_id,
)
from app.providers import (
    GeminiEmbeddingProvider,
    GeminiGenerationProvider,
    ProviderError,
    ProviderTimeout,
)
from app.rag import Answer, RagService
from app.store import CollectionInfo, StoreError, VectorStore

logger = get_logger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question about the documentation.")
    top_k: int | None = Field(
        default=None, description="Chunks to retrieve. Defaults to retrieval.top_k."
    )


class CitationModel(BaseModel):
    source_path: str
    chunk_ordinal: int
    similarity: float
    snippet: str = Field(description="Text of the chunk that was placed in the answer's context.")
    injection_suspected: bool = Field(
        description="True when the chunk contains instruction-like text. Logged, never obeyed."
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    grounded: bool
    correlation_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    correlation_id: str


class IndexStatus(Protocol):
    def ping(self) -> bool: ...
    def collection_info(self) -> CollectionInfo | None: ...


@dataclass(frozen=True)
class Services:
    config: Config
    index: IndexStatus
    rag: RagService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    configure_logging(config.logging.level)

    store = VectorStore(config)
    store.initialise()
    store.verify_embedding_model()

    app.state.services = Services(
        config=config,
        index=store,
        rag=RagService(
            config=config,
            store=store,
            embedder=GeminiEmbeddingProvider(config),
            generator=GeminiGenerationProvider(config),
        ),
    )

    log_event(logger, "service.start", version=__version__, config=config.redacted())
    yield
    log_event(logger, "service.stop", version=__version__)


app = FastAPI(
    title="Documentation Q&A (RAG PoC)",
    version=__version__,
    lifespan=lifespan,
)


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _error(
    status_code: int, error: str, detail: str, correlation_id: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error, detail=detail, correlation_id=correlation_id
        ).model_dump(),
        headers={"X-Correlation-Id": correlation_id},
    )


@app.get("/health")
def health(services: ServicesDep) -> Response:
    reachable = services.index.ping()
    info = services.index.collection_info() if reachable else None
    body: dict[str, Any] = {
        "status": "ok" if reachable else "degraded",
        "version": __version__,
        "vector_database": {
            "reachable": reachable,
            "collection": info.name if info else None,
            "chunk_count": info.chunk_count if info else 0,
            "embedding_model": info.embedding_model if info else None,
        },
    }
    return JSONResponse(
        status_code=status.HTTP_200_OK if reachable else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body,
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
def query(
    request: QueryRequest,
    services: ServicesDep,
    x_correlation_id: Annotated[str | None, Header(alias="X-Correlation-Id")] = None,
) -> Response:
    config = services.config
    with correlation_context(x_correlation_id or new_correlation_id()) as cid:
        question = (request.question or "").strip()
        rejection = _rejection(question, request.top_k, config)
        if rejection:
            error, detail = rejection
            return _error(status.HTTP_400_BAD_REQUEST, error, detail, cid)

        if config.logging.log_question_text:
            log_event(logger, "query.received", question=question, top_k=request.top_k)

        try:
            result = services.rag.answer(question, request.top_k)
        except ProviderTimeout as exc:
            return _upstream_failure(exc, exc.stage, status.HTTP_504_GATEWAY_TIMEOUT, cid)
        except ProviderError as exc:
            return _upstream_failure(exc, exc.stage, status.HTTP_502_BAD_GATEWAY, cid)
        except StoreError as exc:
            return _upstream_failure(exc, "retrieve", status.HTTP_502_BAD_GATEWAY, cid)

        if config.logging.log_answer_text:
            log_event(logger, "query.answered", answer=result.answer)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=_to_response(result, cid).model_dump(),
            headers={"X-Correlation-Id": cid},
        )


def _rejection(question: str, top_k: int | None, config: Config) -> tuple[str, str] | None:
    if not question:
        return "invalid_request", "The 'question' field must be a non-empty string."
    if len(question) > config.runtime.max_question_chars:
        return (
            "question_too_long",
            f"Question exceeds the maximum length of "
            f"{config.runtime.max_question_chars} characters (got {len(question)}).",
        )
    if top_k is not None and not (1 <= top_k <= config.retrieval.max_top_k):
        return (
            "invalid_top_k",
            f"'top_k' must be between 1 and {config.retrieval.max_top_k} (got {top_k}).",
        )
    return None


def _to_response(result: Answer, cid: str) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationModel(
                source_path=c.source_path,
                chunk_ordinal=c.chunk_ordinal,
                similarity=c.similarity,
                snippet=c.snippet,
                injection_suspected=c.injection_suspected,
            )
            for c in result.citations
        ],
        grounded=result.grounded,
        correlation_id=cid,
    )


_UPSTREAM_ERRORS = {
    status.HTTP_504_GATEWAY_TIMEOUT: (
        "upstream_timeout",
        "The upstream model provider did not respond in time.",
    ),
    status.HTTP_502_BAD_GATEWAY: (
        "upstream_error",
        "An upstream dependency failed. The correlation id identifies this request "
        "in the service logs.",
    ),
}


def _upstream_failure(exc: Exception, stage: str, status_code: int, cid: str) -> JSONResponse:
    log_event(
        logger,
        "query.failed",
        level=logging.ERROR,
        stage=stage,
        error_class=type(exc).__name__,
    )
    error, detail = _UPSTREAM_ERRORS[status_code]
    return _error(status_code, error, detail, cid)


@app.exception_handler(RequestValidationError)
def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI answers schema errors with 422; the contract uses 400 for every malformed request."""
    fields = ", ".join(".".join(str(part) for part in err["loc"][1:]) for err in exc.errors())
    return _error(
        status.HTTP_400_BAD_REQUEST,
        "invalid_request",
        f"Malformed request body. Invalid or missing field(s): {fields or 'body'}.",
        new_correlation_id(),
    )
