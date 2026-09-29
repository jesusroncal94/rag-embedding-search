from __future__ import annotations

import json
import time

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from app.config import Config
from app.providers import GeminiGenerationProvider, ProviderError, ProviderTimeout, RetryBudget
from app.rag import INSUFFICIENT_CONTEXT_ANSWER, SOURCES_ONLY_ANSWER
from tests.conftest import (
    FEATURES_DIR,
    LEXICAL_FLOOR,
    OUT_OF_CORPUS,
    REFUSING_FLOOR,
    FakeEmbedder,
    FakeGenaiClient,
    FakeGenerator,
    InMemoryStore,
    seed_corpus,
    wire_ctx,
    with_retrieval,
    with_runtime,
)

pytestmark = pytest.mark.unit

scenarios(str(FEATURES_DIR / "query_api.feature"))

TRUNCATED_ANSWER = "To dispute a transaction, open the"


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


@given(
    parsers.parse(
        'the Gemini generation endpoint does not respond within "runtime.request_timeout_ms"'
    )
)
def _generation_timeout(ctx):
    ctx["generator"] = FakeGenerator(error=ProviderTimeout("generate", "upstream timed out"))


@given("the model stops at the output token limit mid-answer")
def _generation_truncated(ctx):
    client = FakeGenaiClient(answer=TRUNCATED_ANSWER, finish_reason="MAX_TOKENS")
    ctx["generator"] = GeminiGenerationProvider(ctx["config"], client=client)


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


@when(
    parsers.re(
        r'I POST to "/api/v1/query" with question "(?P<question>[^"]*)" and top_k (?P<top_k>\d+)'
    )
)
def _post_with_top_k(ctx, question, top_k):
    _post(ctx, {"question": question, "top_k": int(top_k)})


@when(parsers.parse('I POST to "/api/v1/query" with a question of {size:d} characters'))
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
    stored = {
        (chunk.source_path, chunk.chunk_ordinal): chunk.text for chunk, _ in ctx["store"].rows
    }
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


@then("the response body does not contain the truncated text")
def _truncated_text_withheld(ctx):
    assert TRUNCATED_ANSWER not in ctx["response"].text


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


@given(parsers.re(r'"runtime\.(?P<key>breaker_\w+)" is (?P<value>\d+)'))
def _set_breaker(ctx, key, value):
    ctx["config"] = with_runtime(ctx["config"], **{key: int(value)})


@given("the Gemini generation endpoint fails once and then recovers")
def _generation_recovers(ctx):
    ctx["generator"] = FakeGenerator(failures=[ProviderError("generate", "upstream call failed")])


@given("the Gemini embedding endpoint keeps failing")
def _embedding_down(ctx):
    ctx["embedder"] = FakeEmbedder(
        failures=[ProviderError("embed", "upstream call failed") for _ in range(10)]
    )


def _post_repeatedly(ctx, question: str, times: int, pause_s: float = 0.0) -> None:
    client = wire_ctx(ctx)
    ctx["responses"] = []
    for attempt in range(times):
        if attempt and pause_s:
            time.sleep(pause_s)
        ctx["responses"].append(client.post("/api/v1/query", json={"question": question}))


@when(parsers.parse('I POST the question "{question}" {times:d} times'))
def _post_times(ctx, question, times):
    _post_repeatedly(ctx, question, times)


@when(parsers.parse('I POST the question "{question}", wait {ms:d} ms, and POST it again'))
def _post_after_pause(ctx, question, ms):
    _post_repeatedly(ctx, question, 2, pause_s=ms / 1000)


@then(parsers.parse("the first {n:d} responses are {code:d}"))
def _first_responses(ctx, n, code):
    assert [r.status_code for r in ctx["responses"][:n]] == [code] * n


@then("every response is 502")
def _every_response_502(ctx):
    assert {r.status_code for r in ctx["responses"]} == {502}


@then('the last response is 200 with "generated" false and the retrieved sections as citations')
def _sources_only(ctx):
    last = ctx["responses"][-1]
    assert last.status_code == 200
    body = last.json()
    assert body["generated"] is False
    assert body["answer"] == SOURCES_ONLY_ANSWER
    assert body["citations"]


@then('the last response is 200 with "generated" true')
def _generated_again(ctx):
    last = ctx["responses"][-1]
    assert last.status_code == 200
    assert last.json()["generated"] is True


@then(parsers.parse("the model was called {n:d} times"))
def _model_calls(ctx, n):
    assert ctx["generator"].call_count == n


@then(parsers.parse("the embedding provider was called {n:d} times"))
def _embedding_calls(ctx, n):
    assert ctx["embedder"].attempts == n


@then(parsers.parse('a log line reports that the "{stage}" circuit {state}'))
def _circuit_logged(ctx, stage, state):
    events = ctx["logs"].events(f"provider.circuit_{state}")
    assert [event["stage"] for event in events] == [stage]


class _Transient503(Exception):
    code = 503


@given(parsers.parse("the embedding retry budget holds {tokens:d} tokens"))
def _retry_budget(ctx, tokens):
    ctx["retry_budget"] = RetryBudget(max_tokens=tokens, token_ratio=0.1)


@given("the Gemini embedding endpoint keeps failing with a 503")
def _embedding_503(ctx):
    ctx["embedder"] = FakeEmbedder(
        failures=[_Transient503() for _ in range(20)], budget=ctx["retry_budget"]
    )


@then(parsers.parse('a log line reports that the "{stage}" retry budget is spent'))
def _budget_spent_logged(ctx, stage):
    events = ctx["logs"].events("provider.retry_budget_spent")
    assert events
    assert {event["stage"] for event in events} == {stage}
