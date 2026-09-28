from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from app import ingest as ingest_module
from app.providers import ProviderAuthError
from tests.conftest import FEATURES_DIR, FakeEmbedder, InMemoryStore

pytestmark = pytest.mark.unit

scenarios(str(FEATURES_DIR / "ingestion.feature"))


class _Transient503(Exception):
    code = 503


@pytest.fixture
def ctx(tmp_path, write_config, monkeypatch, log_capture):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    source = tmp_path / "corpus"
    source.mkdir()
    return {
        "source": source,
        "overrides": {},
        "write_config": write_config,
        "monkeypatch": monkeypatch,
        "logs": log_capture,
        "store": InMemoryStore(),
        "embedder": FakeEmbedder(),
        "exit_code": None,
    }


def _write_doc(ctx, name: str, text: str) -> None:
    (ctx["source"] / name).write_text(text, encoding="utf-8")


def _paragraphs(*lengths: int) -> str:
    return "\n\n".join(chr(ord("a") + index % 26) * length for index, length in enumerate(lengths))


def _sentence_text(total: int) -> str:
    body = ""
    while len(body) < total:
        body += "x" * 98 + ". "
    return body[:total].rstrip() + "."


@given(parsers.parse("a configuration file with:"))
def _config_table(ctx, datatable):
    _header, *rows = datatable
    for row in rows:
        key, value = row[0].strip(), row[1].strip()
        ctx["overrides"][key] = int(value) if value.isdigit() else value


@given("a reachable vector database")
def _reachable_db(ctx):
    ctx["store"].reachable = True


@given("a valid Gemini API key in the environment")
def _valid_key(ctx):
    ctx["monkeypatch"].setenv("GEMINI_API_KEY", "test-key-not-real")


@given(parsers.parse('a source directory containing {n:d} markdown documents and 1 ".png" file'))
def _docs_and_png(ctx, n):
    for index in range(n):
        _write_doc(ctx, f"doc{index}.md", _paragraphs(400))
    (ctx["source"] / "image.png").write_bytes(b"\x89PNG\r\n")


@given(
    parsers.parse(
        "a document with {n:d} paragraphs separated by blank lines, each {size:d} characters"
    )
)
def _n_paragraphs(ctx, n, size):
    _write_doc(ctx, "doc.md", _paragraphs(*([size] * n)))


@given(parsers.parse("a document whose paragraphs are {a:d}, {b:d} and {c:d} characters"))
def _three_paragraphs(ctx, a, b, c):
    _write_doc(ctx, "doc.md", _paragraphs(a, b, c))


@given(parsers.parse("a document whose paragraphs are {a:d} and {b:d} characters"))
def _two_paragraphs(ctx, a, b):
    _write_doc(ctx, "doc.md", _paragraphs(a, b))


@given(
    parsers.parse(
        "a document with a single paragraph of {size:d} characters containing sentence terminators"
    )
)
def _one_big_paragraph_with_sentences(ctx, size):
    _write_doc(ctx, "doc.md", _sentence_text(size))


@given(
    parsers.parse(
        "a document with a single {size:d}-character paragraph containing no sentence terminator"
    )
)
def _one_big_paragraph_no_sentences(ctx, size):
    _write_doc(ctx, "doc.md", "a" * size)


@given(parsers.parse("a source directory containing 1 document of {n:d} paragraphs"))
def _one_doc_n_paragraphs(ctx, n):
    _write_doc(ctx, "doc.md", _paragraphs(*([400] * n)))


@given(parsers.parse("a corpus that produces {n:d} chunks"))
def _corpus_of_n_chunks(ctx, n):
    for index in range(n):
        _write_doc(ctx, f"doc{index:03d}.md", "b" * 400)


@given(parsers.parse('"embedding.embed_batch_size" is {size:d}'))
def _batch_size(ctx, size):
    ctx["overrides"]["embedding.embed_batch_size"] = size


@given(
    parsers.parse(
        "the embedding API fails with a 503 on the first {n:d} attempts and succeeds on the third"
    )
)
def _transient_failures(ctx, n):
    _write_doc(ctx, "doc.md", "c" * 400)
    ctx["embedder"] = FakeEmbedder(failures=[_Transient503() for _ in range(n)])


@given("the embedding API fails with a 401 on every attempt")
def _permanent_failure(ctx):
    _write_doc(ctx, "doc.md", "c" * 400)
    ctx["embedder"] = FakeEmbedder(
        failures=[ProviderAuthError("embed", "credentials rejected (HTTP 401)") for _ in range(10)]
    )


@given("the ingestion script has already been run once over the corpus")
def _already_ingested(ctx):
    for index in range(3):
        _write_doc(ctx, f"doc{index}.md", "d" * 400)
    _run_ingest(ctx)
    ctx["first_count"] = len(ctx["store"].rows)


def _run_ingest(ctx) -> None:
    if not any(ctx["source"].iterdir()):
        _write_doc(ctx, "doc.md", "e" * 400)

    config_path = ctx["write_config"](ctx["overrides"])
    embedder = ctx["embedder"]
    store = ctx["store"]

    ctx["monkeypatch"].setattr(ingest_module, "GeminiEmbeddingProvider", lambda config: embedder)
    ctx["monkeypatch"].setattr(ingest_module, "VectorStore", lambda config: store)

    ctx["exit_code"] = ingest_module.main(
        ["--config", str(config_path), "--source", str(ctx["source"])]
    )


@when("I run the ingestion script")
def _run(ctx):
    _run_ingest(ctx)


@when("I run the ingestion script a second time over the same corpus")
def _run_again(ctx):
    _run_ingest(ctx)


def _summary(ctx) -> dict:
    events = ctx["logs"].events("ingest.complete")
    assert events, "no ingest.complete summary was emitted"
    return events[-1]


@then(parsers.parse("{n:d} documents are reported as processed"))
def _documents_processed(ctx, n):
    assert _summary(ctx)["documents_processed"] == n


@then(parsers.parse('the ".png" file is skipped without failing the run'))
def _png_skipped(ctx):
    skipped = ctx["logs"].events("ingest.file_skipped")
    assert any(".png" in event["source_path"] for event in skipped)
    assert ctx["exit_code"] == 0


@then(parsers.re(r"^(?P<n>\d+) chunks? (?:are|is) stored$"))
def _chunks_stored(ctx, n):
    assert len(ctx["store"].rows) == int(n)


@then("no chunk contains text from more than one paragraph")
def _no_mixed_paragraphs(ctx):
    for chunk, _ in ctx["store"].rows:
        assert len(set(chunk.text)) == 1


@then("that chunk contains the text of all 3 paragraphs in original order")
def _merged_in_order(ctx):
    text = ctx["store"].rows[0][0].text
    positions = [text.find(char) for char in "abc"]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


@then(parsers.parse("the {size:d}-character chunk is stored rather than discarded"))
def _short_chunk_kept(ctx, size):
    assert any(len(chunk.text) == size for chunk, _ in ctx["store"].rows)


@then(parsers.parse("every chunk is at most {size:d} characters"))
def _chunk_size_limit(ctx, size):
    assert all(len(chunk.text) <= size for chunk, _ in ctx["store"].rows)


@then("every chunk ends at a sentence terminator")
def _ends_at_sentence(ctx):
    assert all(chunk.text.rstrip().endswith(".") for chunk, _ in ctx["store"].rows)


@then(parsers.parse('chunks are produced by hard character split at "max_chunk_chars"'))
def _hard_split(ctx):
    limit = ctx["overrides"].get("chunking.max_chunk_chars", 1500)
    assert all(len(chunk.text) <= limit for chunk, _ in ctx["store"].rows)
    assert any(len(chunk.text) == limit for chunk, _ in ctx["store"].rows)


@then("a warning is logged naming the source document")
def _hard_split_warning(ctx):
    events = ctx["logs"].events("ingest.hard_split")
    assert events
    assert "doc.md" in events[0]["detail"]


@then("every stored chunk has a non-empty embedding vector")
def _has_vectors(ctx):
    assert ctx["store"].rows
    assert all(vector and any(vector) for _, vector in ctx["store"].rows)


@then(
    parsers.parse(
        'every stored chunk carries "source_path", "chunk_ordinal", "start_offset" and "end_offset"'
    )
)
def _has_metadata(ctx):
    for chunk, _ in ctx["store"].rows:
        assert chunk.source_path
        assert chunk.chunk_ordinal >= 0
        assert chunk.start_offset >= 0
        assert chunk.end_offset > chunk.start_offset


@then("the chunk ordinals are contiguous starting at 0")
def _contiguous_ordinals(ctx):
    ordinals = [chunk.chunk_ordinal for chunk, _ in ctx["store"].rows]
    assert ordinals == list(range(len(ordinals)))


@then(parsers.parse("exactly {n:d} embedding API calls are made"))
def _embedding_calls(ctx, n):
    assert ctx["embedder"].call_count == n
    assert _summary(ctx)["embedding_calls"] == n


@then("ingestion completes successfully")
def _ingest_ok(ctx):
    assert ctx["exit_code"] == 0


@then(parsers.parse("{n:d} retry events are logged with increasing backoff"))
def _retries_logged(ctx, n):
    retries = ctx["logs"].events("provider.retry")
    assert len(retries) == n
    backoffs = [event["backoff_ms"] for event in retries]
    assert backoffs == sorted(backoffs)
    assert len(set(backoffs)) == len(backoffs), "backoff must increase, not repeat"


@then("the script exits with a non-zero status")
def _nonzero_exit(ctx):
    assert ctx["exit_code"] not in (None, 0)


@then(parsers.parse('a structured error log names the failing stage as "{stage}"'))
def _error_stage(ctx, stage):
    failures = ctx["logs"].events("ingest.failed")
    assert failures
    assert failures[-1]["stage"] == stage


@then("no partially-embedded collection is left serving queries")
def _no_partial_collection(ctx):
    assert ctx["store"].replace_calls == 0
    assert ctx["store"].rows == []


@then("the stored chunk count is unchanged")
def _count_unchanged(ctx):
    assert len(ctx["store"].rows) == ctx["first_count"]


@then("no chunk is duplicated")
def _no_duplicates(ctx):
    keys = [(chunk.source_path, chunk.chunk_ordinal) for chunk, _ in ctx["store"].rows]
    assert len(keys) == len(set(keys))


@then(
    parsers.parse(
        'a single JSON log line reports "documents_processed", "chunks_created", '
        '"embedding_calls" and "duration_ms"'
    )
)
def _summary_fields(ctx):
    events = ctx["logs"].events("ingest.complete")
    assert len(events) == 1
    for field in ("documents_processed", "chunks_created", "embedding_calls", "duration_ms"):
        assert field in events[0], f"missing {field}"


@then("the collection metadata records the embedding model name and dimensionality")
def _collection_metadata(ctx):
    info = ctx["store"].collection_info()
    assert info is not None
    assert info.embedding_model
    assert info.dimensions > 0
