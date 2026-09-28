from __future__ import annotations

import json

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from app.config import Config
from app.providers import ProviderError, ProviderTimeout
from app.rag import INSUFFICIENT_CONTEXT_ANSWER
from tests.conftest import (
    FEATURES_DIR,
    LEXICAL_FLOOR,
    OUT_OF_CORPUS,
    REFUSING_FLOOR,
    FakeEmbedder,
    FakeGenerator,
    InMemoryStore,
    seed_corpus,
    wire_ctx,
    with_retrieval,
)

pytestmark = pytest.mark.unit

scenarios(str(FEATURES_DIR / "query_api.feature"))


@pytest.fixture
def ctx(base_config: Config, log_capture):
    return {
        "config": base_config,
        "store": InMemoryStore(),
        "embedder": FakeEmbedder(),
        "generator": FakeGenerator(answer="A grounded answer. [payments-and-transfers.md]"),
        "logs": log_capture,
        "response": None,
    }


@given("the corpus has been ingested")
def _corpus_ingested(ctx):
    seed_corpus(ctx["store"], ctx["embedder"], ctx["config"])


@given("the API service is healthy")
def _service_healthy(ctx):
    ctx["store"].reachable = True


@given(parsers.parse('"retrieval.top_k" is {value:d}'))
def _set_top_k(ctx, value):
    ctx["config"] = with_retrieval(ctx["config"], top_k=value)


@given(parsers.parse('"retrieval.min_similarity" is {value:f}'))
def _set_min_similarity(ctx, value):
    ctx["config"] = with_retrieval(ctx["config"], min_similarity=LEXICAL_FLOOR)


@given("the Gemini generation endpoint returns a 500")
def _generation_500(ctx):
    ctx["generator"] = FakeGenerator(error=ProviderError("generate", "upstream call failed"))


@given(parsers.parse('the Gemini generation endpoint does not respond within "runtime.request_timeout_ms"'))
def _generation_timeout(ctx):
    ctx["generator"] = FakeGenerator(error=ProviderTimeout("generate", "upstream timed out"))


@given("the vector database is unreachable")
def _db_unreachable(ctx):
    ctx["store"].reachable = False


def _post(ctx, payload: dict, headers: dict | None = None) -> None:
    if payload.get("question") in OUT_OF_CORPUS:
        ctx["config"] = with_retrieval(ctx["config"], min_similarity=REFUSING_FLOOR)
    ctx["response"] = wire_ctx(ctx).post("/api/v1/query", json=payload, headers=headers or {})


@when(parsers.re(r'I POST to "/api/v1/query" with question "(?P<question>[^"]*)"$'))
def _post_question(ctx, question):
    _post(ctx, {"question": question})


@when(parsers.re(r'I POST to "/api/v1/query" with question (?P<question>omitted)$'))
def _post_omitted(ctx, question):
    _post(ctx, {})


@when(parsers.re(r'I POST to "/api/v1/query" with question "(?P<question>[^"]*)" and top_k (?P<top_k>\d+)'))
def _post_with_top_k(ctx, question, top_k):
    _post(ctx, {"question": question, "top_k": int(top_k)})


@when(parsers.parse("I POST to \"/api/v1/query\" with a question of {size:d} characters"))
def _post_long_question(ctx, size):
    _post(ctx, {"question": "a" * size})


@when(parsers.parse('I POST to "/api/v1/query" with header "X-Correlation-Id" set to "{value}"'))
def _post_with_correlation_id(ctx, value):
    _post(ctx, {"question": "How do I deploy?"}, headers={"X-Correlation-Id": value})


@when(parsers.parse('I GET "/health"'))
def _get_health(ctx):
    ctx["response"] = wire_ctx(ctx).get("/health")


def _body(ctx) -> dict:
    return ctx["response"].json()


@then(parsers.parse("the response status is {code:d}"))
def _status(ctx, code):
    assert ctx["response"].status_code == code, _body(ctx)


@then(parsers.parse('the response body contains a non-empty "answer"'))
def _non_empty_answer(ctx):
    assert _body(ctx)["answer"].strip()


@then(parsers.parse("the response body contains at least {n:d} citation"))
def _min_citations(ctx, n):
    assert len(_body(ctx)["citations"]) >= n


@then(parsers.parse('every citation carries "source_path" and "chunk_ordinal"'))
def _citation_fields(ctx):
    for citation in _body(ctx)["citations"]:
        assert citation["source_path"]
        assert citation["chunk_ordinal"] >= 0


@then("every citation carries the text of the chunk it cites")
def _citation_snippets(ctx):
    stored = {(chunk.source_path, chunk.chunk_ordinal): chunk.text for chunk, _ in ctx["store"].rows}
    citations = _body(ctx)["citations"]
    assert citations
    for citation in citations:
        assert citation["snippet"] == stored[(citation["source_path"], citation["chunk_ordinal"])]


@then("every citation says whether its chunk looked like an injection")
def _citation_flags(ctx):
    for citation in _body(ctx)["citations"]:
        assert isinstance(citation["injection_suspected"], bool)


@then("every citation corresponds to a chunk that was placed in the LLM context")
def _citations_match_context(ctx):
    prompt = ctx["generator"].prompts[-1]
    for citation in _body(ctx)["citations"]:
        marker = f'source="{citation["source_path"]}" chunk="{citation["chunk_ordinal"]}"'
        assert marker in prompt, f"citation not present in prompt: {marker}"


@then(parsers.parse("exactly {n:d} chunks are retrieved"))
def _retrieved_count(ctx, n):
    assert len(_body(ctx)["citations"]) <= n
    events = ctx["logs"].events("retrieval.result")
    assert events[-1]["chunks_retrieved"] == n


@then("the error message names the permitted range")
def _names_range(ctx):
    assert str(ctx["config"].retrieval.max_top_k) in _body(ctx)["detail"]


@then("the error message names the maximum question length")
def _names_max_length(ctx):
    assert str(ctx["config"].runtime.max_question_chars) in _body(ctx)["detail"]


@then(parsers.parse('the "answer" states that the documentation does not contain the information'))
def _refusal(ctx):
    assert _body(ctx)["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert _body(ctx)["grounded"] is False


@then("the citation list is empty")
def _no_citations(ctx):
    assert _body(ctx)["citations"] == []


@then(parsers.re(r"no generation call is made(?: to the LLM)?$"))
def _no_generation(ctx):
    assert ctx["generator"].call_count == 0


@then("no embedding call is made")
def _no_embedding(ctx):
    assert len(ctx["embedder"].query_calls) == 0


@then(parsers.parse('the response body contains a "correlation_id"'))
def _has_correlation_id(ctx):
    assert _body(ctx)["correlation_id"]


@then("the response body contains no stack trace and no API key")
def _no_leakage(ctx):
    raw = json.dumps(_body(ctx))
    assert "Traceback" not in raw
    assert 'File "' not in raw
    assert ctx["config"].gemini_api_key not in raw


@then(parsers.parse('the failure is logged with stage "{stage}"'))
def _logged_stage(ctx, stage):
    failures = ctx["logs"].events("query.failed")
    assert failures
    assert failures[-1]["stage"] == stage


@then("the body reports the vector database as reachable")
def _db_reachable(ctx):
    assert _body(ctx)["vector_database"]["reachable"] is True


@then(parsers.parse('the response carries a "correlation_id"'))
def _response_correlation_id(ctx):
    assert _body(ctx)["correlation_id"]
    assert ctx["response"].headers.get("X-Correlation-Id")


@then(parsers.parse('every log line emitted for that request carries the same "correlation_id"'))
def _logs_share_correlation_id(ctx):
    cid = _body(ctx)["correlation_id"]
    request_events = {"query.received", "retrieval.result", "embedding.latency", "llm.latency"}
    seen = [r for r in ctx["logs"].records if r.get("event") in request_events]
    assert seen, "no request-scoped log lines were emitted"
    for record in seen:
        assert record.get("correlation_id") == cid


@then(parsers.parse('the response "correlation_id" is "{value}"'))
def _correlation_id_is(ctx, value):
    assert _body(ctx)["correlation_id"] == value
