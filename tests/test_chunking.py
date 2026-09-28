from __future__ import annotations

from itertools import pairwise

import pytest

from app.chunking import Chunk, chunk_text, discover_documents

MIN_CHARS = 200
MAX_CHARS = 1500


def _chunk(text: str, *, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS):
    return chunk_text(
        text,
        source_path="doc.md",
        min_chunk_chars=min_chars,
        max_chunk_chars=max_chars,
    )


def _paragraph(length: int, filler: str = "a") -> str:
    return filler * length


def _sentences(total: int, sentence_len: int = 100) -> str:
    body = ""
    while len(body) < total:
        body += "x" * (sentence_len - 2) + ". "
    return body[:total].rstrip() + "."


pytestmark = pytest.mark.unit


def test_splits_on_paragraph_boundaries():
    text = "\n\n".join(_paragraph(400, c) for c in "abc")
    chunks, _ = _chunk(text)
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(set(chunk.text)) == 1


def test_collapses_repeated_delimiters():
    text = _paragraph(300, "a") + "\n\n\n\n" + _paragraph(300, "b")
    chunks, _ = _chunk(text)
    assert len(chunks) == 2
    assert chunks[0].text.strip() == _paragraph(300, "a")


def test_empty_document_yields_no_chunks():
    assert _chunk("")[0] == []
    assert _chunk("\n\n   \n\n")[0] == []


def test_short_paragraphs_merge_forward():
    text = "\n\n".join([_paragraph(50, "a"), _paragraph(60, "b"), _paragraph(900, "c")])
    chunks, _ = _chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text.index("a") < chunks[0].text.index("b") < chunks[0].text.index("c")


def test_trailing_short_paragraph_is_kept_not_discarded():
    text = _paragraph(900, "a") + "\n\n" + _paragraph(50, "b")
    chunks, _ = _chunk(text)
    assert len(chunks) == 2
    assert chunks[1].text == _paragraph(50, "b")
    assert len(chunks[1].text) == 50


def test_document_shorter_than_minimum_is_still_stored():
    chunks, _ = _chunk(_paragraph(10, "a"))
    assert len(chunks) == 1
    assert chunks[0].text == _paragraph(10, "a")


def test_oversized_paragraph_splits_at_sentence_boundaries():
    chunks, warnings = _chunk(_sentences(4000))
    assert len(chunks) == 3
    assert all(len(chunk.text) <= MAX_CHARS for chunk in chunks)
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)
    assert warnings == []


def test_oversized_paragraph_without_sentence_boundary_is_hard_split():
    chunks, warnings = _chunk(_paragraph(4000, "a"))
    assert all(len(chunk.text) <= MAX_CHARS for chunk in chunks)
    assert sum(len(chunk.text) for chunk in chunks) == 4000
    assert warnings, "a hard split must be reported"
    assert "doc.md" in warnings[0]


def test_abbreviations_do_not_trigger_a_sentence_split():
    text = "See e.g. the policy. " * 200
    chunks, _ = _chunk(text[:3000])
    for chunk in chunks:
        assert not chunk.text.rstrip().endswith("e.g.")


def test_merge_then_split_never_leaks_an_oversized_chunk():
    text = "\n\n".join(_paragraph(150, "a") for _ in range(40))
    chunks, _ = _chunk(text)
    assert all(len(chunk.text) <= MAX_CHARS for chunk in chunks)


def test_metadata_is_complete_and_offsets_resolve():
    text = "\n\n".join(_paragraph(400, c) for c in "abcde")
    chunks, _ = _chunk(text)
    assert [chunk.chunk_ordinal for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.source_path == "doc.md"
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text


def test_offsets_are_non_overlapping_and_ascending():
    text = "\n\n".join(_paragraph(400, c) for c in "abcd")
    chunks, _ = _chunk(text)
    for earlier, later in pairwise(chunks):
        assert earlier.end_offset <= later.start_offset


def test_chunk_rejects_inverted_offsets():
    with pytest.raises(ValueError):
        Chunk(text="x", source_path="d.md", chunk_ordinal=0, start_offset=5, end_offset=1)


def test_real_corpus_chunks_cleanly(corpus_dir):
    total = 0
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks, warnings = _chunk(text)
        assert chunks, f"{path.name} produced no chunks"
        assert warnings == [], f"{path.name} required a hard split"
        for chunk in chunks:
            assert text[chunk.start_offset : chunk.end_offset] == chunk.text
            assert len(chunk.text) <= MAX_CHARS
        total += len(chunks)
    assert total > 0


def test_injection_payload_survives_chunking_intact(corpus_dir):
    """If chunking split the payload, the @live injection scenarios would test nothing."""
    text = (corpus_dir / "account-and-verification.md").read_text(encoding="utf-8")
    chunks, _ = _chunk(text)
    assert any("Ignore all previous instructions" in chunk.text for chunk in chunks)


def test_discover_documents_skips_unsupported_extensions(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    supported, skipped = discover_documents(tmp_path, [".md", ".txt"])
    assert {p.name for p in supported} == {"a.md", "b.txt"}
    assert {p.name for p in skipped} == {"c.png"}


def test_discover_documents_raises_on_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_documents(tmp_path / "nope", [".md"])
