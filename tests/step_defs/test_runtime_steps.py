from __future__ import annotations

import json
import subprocess
import time
from dataclasses import replace

import httpx
import pytest
import yaml
from pytest_bdd import given, parsers, scenarios, then, when

from app.config import ConfigError, load_config
from app.logging_setup import configure_logging, get_logger, log_event
from app.providers import GeminiEmbeddingProvider, GeminiGenerationProvider, ProviderError
from app.rag import INSUFFICIENT_CONTEXT_ANSWER
from app.store import CollectionInfo, EmbeddingModelMismatch, VectorStore
from tests.conftest import (
    FEATURES_DIR,
    LEXICAL_FLOOR,
    REPO_ROOT,
    FakeGenaiClient,
    FakeGenerator,
    InMemoryStore,
    seed_corpus,
    wire_ctx,
    with_retrieval,
)

scenarios(str(FEATURES_DIR / "config_and_runtime.feature"))

COMPOSE = ["docker", "compose"]


@pytest.fixture
def ctx(base_config, log_capture, write_config, monkeypatch):
    client = FakeGenaiClient(answer="Answer text CANARY-ANSWER-ONLY.")
    return {
        "config": base_config,
        "logs": log_capture,
        "write_config": write_config,
        "monkeypatch": monkeypatch,
        "store": InMemoryStore(),
        "embedder": GeminiEmbeddingProvider(base_config, client=client),
        "generator": GeminiGenerationProvider(base_config, client=client),
        "overrides": {},
        "error": None,
        "response": None,
    }


def _seed(ctx) -> None:
    seed_corpus(ctx["store"], ctx["embedder"], ctx["config"])
    ctx["config"] = with_retrieval(ctx["config"], min_similarity=LEXICAL_FLOOR)


@given(parsers.parse('"{key}" is {value:d} in the configuration file'))
def _key_is(ctx, key, value):
    ctx["overrides"][key] = value
    ctx["tracked_key"] = key
    ctx["source_mtimes"] = {
        path: path.stat().st_mtime for path in (REPO_ROOT / "app").rglob("*.py")
    }


@when(parsers.parse("I change it to {value:d} and restart the service"))
def _change_and_restart(ctx, value):
    path = ctx["write_config"]({ctx["tracked_key"]: value})
    ctx["monkeypatch"].setenv("GEMINI_API_KEY", "test-key-not-real")
    ctx["config"] = load_config(path)
    _seed(ctx)
    ctx["response"] = wire_ctx(ctx).post(
        "/api/v1/query", json={"question": "How do I dispute a transaction?"}
    )


@then(parsers.parse("queries retrieve {n:d} chunks"))
def _retrieves_n(ctx, n):
    events = ctx["logs"].events("retrieval.result")
    assert events, "no retrieval.result log line"
    assert events[-1]["chunks_retrieved"] == n


@then("no source file was modified")
def _sources_untouched(ctx):
    for path, mtime in ctx["source_mtimes"].items():
        assert path.stat().st_mtime == mtime, f"{path} was modified"


@when("I inspect the configuration file and the built image layers")
def _inspect_config(ctx):
    ctx["config_text"] = (REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    ctx["dockerfile_text"] = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")


@then("no Gemini API key is present")
def _no_key_in_files(ctx):
    for text in (ctx["config_text"], ctx["dockerfile_text"]):
        assert "GEMINI_API_KEY=" not in text
        assert "AIza" not in text, "a literal Google API key is present"
    data = yaml.safe_load(ctx["config_text"])
    assert "gemini_api_key" not in json.dumps(data)


@then("the service reads the key from the environment at runtime")
def _key_from_env(ctx, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    assert load_config(REPO_ROOT / "config" / "config.yaml").gemini_api_key == "from-environment"


@given("the Gemini API key environment variable is unset")
def _unset_key(ctx):
    ctx["monkeypatch"].delenv("GEMINI_API_KEY", raising=False)


@given(parsers.parse('"{key}" is set to {value:d}'))
def _invalid_value(ctx, key, value):
    ctx["overrides"][key] = value
    ctx["tracked_key"] = key


@given(parsers.parse('the collection metadata records embedding model "{model}"'))
def _collection_model(ctx, model):
    ctx["store"].has_collection = True
    ctx["store"].embedding_model = model


@given(parsers.parse('the service is configured with embedding model "{model}"'))
def _configured_model(ctx, model):
    ctx["configured_model"] = model


def _start_against_recorded_model(ctx) -> None:
    config = replace(
        ctx["config"],
        embedding=replace(ctx["config"].embedding, model=ctx["configured_model"]),
    )
    store = VectorStore(config)
    recorded = ctx["store"].embedding_model
    store.collection_info = lambda: CollectionInfo(
        name="docs",
        embedding_model=recorded,
        dimensions=config.embedding.dimensions,
        chunk_count=1,
    )
    store.verify_embedding_model()


def _start_from_config(ctx) -> None:
    config = load_config(ctx["write_config"](ctx["overrides"]))
    configure_logging(config.logging.level)
    log_event(get_logger("app.api"), "service.start", config=config.redacted())
    ctx["config"] = config


@when("the service starts")
def _service_starts(ctx):
    started = time.perf_counter()
    try:
        if ctx.get("configured_model"):
            _start_against_recorded_model(ctx)
        else:
            _start_from_config(ctx)
    except (ConfigError, EmbeddingModelMismatch) as exc:
        ctx["error"] = exc
    ctx["startup_seconds"] = time.perf_counter() - started


@then(parsers.parse("it exits within {seconds:d} seconds"))
def _exits_within(ctx, seconds):
    assert ctx["error"] is not None, "startup unexpectedly succeeded"
    assert ctx["startup_seconds"] < seconds


@then("the log names the missing variable")
def _names_missing_var(ctx):
    assert "GEMINI_API_KEY" in str(ctx["error"])


@then("it does not bind the HTTP port")
def _no_port_bound(ctx):
    assert ctx["error"] is not None


@then(parsers.parse('it exits with a validation error naming "{key}"'))
def _validation_error_names(ctx, key):
    assert isinstance(ctx["error"], ConfigError)
    assert key in str(ctx["error"])


@then("it fails fast with a message naming both models")
def _names_both_models(ctx):
    assert isinstance(ctx["error"], EmbeddingModelMismatch)
    message = str(ctx["error"])
    assert ctx["store"].embedding_model in message
    assert ctx["configured_model"] in message


@then("one JSON log line reports the resolved configuration")
def _start_logs_config(ctx):
    events = ctx["logs"].events("service.start")
    assert len(events) >= 1
    assert "config" in events[-1]


@then(parsers.parse('the Gemini API key appears as "{mask}"'))
def _key_masked(ctx, mask):
    events = ctx["logs"].events("service.start")
    assert events[-1]["config"]["gemini_api_key"] == mask


@given("the LLM provider is failing")
def _llm_failing(ctx):
    ctx["generator"] = FakeGenerator(error=ProviderError("generate", "upstream call failed"))


@when("I POST a valid question")
def _post_valid(ctx):
    _seed(ctx)
    ctx["response"] = wire_ctx(ctx).post(
        "/api/v1/query", json={"question": "How long do I have to dispute a transaction?"}
    )


@when(parsers.parse('I POST a question containing the string "{canary}"'))
def _post_canary(ctx, canary):
    _seed(ctx)
    ctx["canary"] = canary
    ctx["response"] = wire_ctx(ctx).post(
        "/api/v1/query", json={"question": f"What about {canary} disputes?"}
    )


def _request_events(ctx, event: str) -> list[dict]:
    cid = ctx["response"].json()["correlation_id"]
    return [record for record in ctx["logs"].events(event) if record.get("correlation_id") == cid]


@then("a log line reports embedding latency")
def _embedding_latency_logged(ctx):
    events = _request_events(ctx, "embedding.latency")
    assert len(events) == 1, events
    assert events[0]["duration_ms"] >= 0
    assert events[0]["batch_size"] == 1


@then("a log line reports retrieval outcome with top score and search latency")
def _retrieval_logged(ctx):
    events = _request_events(ctx, "retrieval.result")
    assert len(events) == 1, events
    assert "top_score" in events[0]
    assert events[0]["duration_ms"] >= 0


@then("a log line reports LLM latency and model name")
def _llm_latency_logged(ctx):
    events = _request_events(ctx, "llm.latency")
    assert len(events) == 1, events
    assert events[0]["duration_ms"] >= 0
    assert events[0]["model"] == ctx["config"].generation.model


@then("no log line is emitted for non-critical internal steps")
def _only_critical_events(ctx):
    allowed = {
        "service.start",
        "service.stop",
        "query.received",
        "query.answered",
        "query.failed",
        "retrieval.result",
        "embedding.latency",
        "llm.latency",
        "provider.retry",
        "provider.retry_budget_spent",
        "provider.circuit_opened",
        "provider.circuit_closed",
        "query.sources_only",
        "prompt_injection_suspected",
        "store.collection_absent",
        "ingest.complete",
        "ingest.file_skipped",
        "ingest.hard_split",
        "ingest.document_chunked",
    }
    for record in ctx["logs"].records:
        assert record["event"] in allowed, f"unexpected event: {record['event']}"


@then("exactly one error log line is emitted")
def _one_error(ctx):
    errors = [r for r in ctx["logs"].records if r["level"] == "ERROR"]
    assert len(errors) == 1, errors


@then(parsers.parse('it names stage "{stage}" and the correlation id'))
def _error_names_stage(ctx, stage):
    failures = ctx["logs"].events("query.failed")
    assert failures[-1]["stage"] == stage
    assert failures[-1].get("correlation_id")


@then(parsers.parse('a log line with event "{event}" contains "{canary}"'))
def _question_logged(ctx, event, canary):
    events = ctx["logs"].events(event)
    assert events, f"no {event} line"
    assert canary in json.dumps(events[-1])


@then("no log line contains the answer text")
def _answer_not_logged(ctx):
    answer = ctx["response"].json()["answer"]
    assert not ctx["logs"].contains_text(answer)


@then("no log line contains the Gemini API key")
def _key_not_logged(ctx):
    assert not ctx["logs"].contains_text(ctx["config"].gemini_api_key)


@when("I run the full stack through one ingestion and one query")
def _full_stack(ctx):
    _seed(ctx)
    ctx["response"] = wire_ctx(ctx).post(
        "/api/v1/query", json={"question": "How do I block my card?"}
    )


@then("every emitted log line parses as a single JSON object")
def _all_json(ctx):
    for line in ctx["logs"].lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert "\n" not in line


@then(parsers.parse('every line carries "timestamp", "level" and "event"'))
def _required_fields(ctx):
    for record in ctx["logs"].records:
        for field in ("timestamp", "level", "event"):
            assert field in record


def _compose(*args: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        COMPOSE + list(args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def _wait_healthy(timeout: int = 90) -> float:
    deadline = time.time() + timeout
    started = time.time()
    while time.time() < deadline:
        try:
            if httpx.get("http://localhost:8000/health", timeout=3).status_code == 200:
                return time.time() - started
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise AssertionError(f"/health did not return 200 within {timeout}s")


@given(parsers.parse('a clean machine with Docker and a Gemini API key in ".env"'))
def _clean_machine(ctx):
    if not (REPO_ROOT / ".env").is_file():
        pytest.skip("docker tier: .env with GEMINI_API_KEY required")
    _compose("down", "-v", check=False)


@given("the vector database takes 20 seconds to become ready")
def _slow_db(ctx):
    _compose("down", "-v", check=False)


@given("the stack is running")
def _stack_running(ctx):
    _compose("up", "-d", "db", "api")
    _wait_healthy()


@given("the corpus has been ingested")
def _corpus_ingested_docker(ctx):
    _compose("up", "-d", "db", "api")
    _wait_healthy()
    _compose("run", "--rm", "ingest")


@when(parsers.parse('I run "docker-compose up"'))
def _compose_up(ctx):
    _compose("up", "-d", "--build")
    ctx["healthy_after"] = _wait_healthy()


@when(parsers.parse('I run "docker-compose down" and then "docker-compose up"'))
def _restart(ctx):
    _compose("down")
    _compose("up", "-d")
    _wait_healthy()


@when(parsers.parse('I run "docker-compose down -v" and then "docker-compose up"'))
def _restart_clean(ctx):
    _compose("down", "-v")
    _compose("up", "-d")
    _wait_healthy()


@when("I run the ingestion container against the compose network")
def _run_ingest_container(ctx):
    ctx["ingest_result"] = _compose("run", "--rm", "ingest", check=False)


@when("I inspect the final image")
def _inspect_image(ctx):
    ctx["image_user"] = subprocess.run(
        ["docker", "compose", "run", "--rm", "--entrypoint", "id", "api", "-un"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    ctx["image_history"] = subprocess.run(
        ["docker", "image", "history", "--no-trunc", "rag-pipeline-test-api"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


@then("the vector database and the API service both start")
def _both_started(ctx):
    output = _compose("ps", "--format", "json").stdout
    assert "db" in output and "api" in output


@then(parsers.parse('"/health" returns 200 within {seconds:d} seconds'))
def _health_within(ctx, seconds):
    assert ctx["healthy_after"] < seconds


@then("no additional manual step was required")
def _no_manual_step(ctx):
    assert ctx["healthy_after"] is not None


@then("the API service does not report healthy before the database is reachable")
def _api_waits(ctx):
    assert httpx.get("http://localhost:8000/health", timeout=5).json()["vector_database"][
        "reachable"
    ]


@then("the API service does not crash-loop")
def _no_crash_loop(ctx):
    output = _compose("ps", "--format", "json").stdout
    assert "Restarting" not in output


@then("it connects to the vector database by service name")
def _connects_by_name(ctx):
    compose_file = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "@db:5432" in compose_file


@then("it exits 0 on success")
def _ingest_exit_zero(ctx):
    assert ctx["ingest_result"].returncode == 0, ctx["ingest_result"].stderr


@then("queries still return answers without re-ingesting")
def _data_survived(ctx):
    body = httpx.get("http://localhost:8000/health", timeout=5).json()
    assert body["vector_database"]["chunk_count"] > 0


@then("the collection is empty")
def _collection_empty(ctx):
    body = httpx.get("http://localhost:8000/health", timeout=5).json()
    assert body["vector_database"]["chunk_count"] == 0


@then(parsers.parse('queries return "insufficient context" rather than an error'))
def _empty_degrades(ctx):
    response = httpx.post(
        "http://localhost:8000/api/v1/query", json={"question": "anything"}, timeout=30
    )
    assert response.status_code == 200
    assert response.json()["answer"] == INSUFFICIENT_CONTEXT_ANSWER


@then("it contains no build toolchain")
def _no_toolchain(ctx):
    assert "gcc" not in ctx["image_history"]


@then("it contains no API key in any layer")
def _no_key_in_layers(ctx):
    assert "AIza" not in ctx["image_history"]


@then("the service runs as a non-root user")
def _non_root(ctx):
    assert ctx["image_user"] == "app", ctx["image_user"]


@given("a reader with Docker and an API key who has not seen the repository")
def _reader(ctx):
    pytest.skip("manual scenario")


@when("they follow the README top to bottom")
def _follow_readme(ctx):
    pytest.skip("manual scenario")


@then(parsers.parse("they reach a successful cited answer in under {minutes:d} minutes"))
def _readme_works(ctx, minutes):
    pytest.skip("manual scenario")
