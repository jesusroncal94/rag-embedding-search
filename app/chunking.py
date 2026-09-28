"""Paragraph-based chunking — the specified strategy (FR-ING-02..04a, FR-ING-06).

Shared by ingestion and (for offset resolution) the query path, which is the whole
reason ADR-001 chose a single language: this logic must never exist twice.

The strategy is deliberately implemented as specified, including its weaknesses.
Known limitation (R-01): paragraph boundaries fragment markdown tables, fenced code
blocks and multi-step procedures, because each row / line is its own paragraph after a
blank-line split. That is a real retrieval weakness on exactly the content users ask
about, it is demonstrated rather than hidden, and structure-aware chunking is the first
post-PoC change proposed in SPEC.md §6.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A sentence terminator followed by whitespace. Used to find split points inside an
# oversized paragraph (FR-ING-04).
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s)")

# Abbreviations that end in a period but do not end a sentence. Small and pragmatic:
# a full solution needs a sentence tokeniser, which is not warranted at PoC scope.
_ABBREVIATIONS = frozenset(
    {
        "e.g", "i.e", "etc", "vs", "cf", "approx", "no", "mr", "mrs", "ms", "dr", "prof",
        "inc", "ltd", "co", "st", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec",
    }
)


@dataclass(frozen=True)
class Chunk:
    """One unit of retrievable text with full provenance (FR-ING-06)."""

    text: str
    source_path: str
    chunk_ordinal: int
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        if self.start_offset > self.end_offset:
            raise ValueError("start_offset must not exceed end_offset")


@dataclass(frozen=True)
class _Span:
    """A region of the source document, by character offset."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _paragraph_spans(text: str, delimiter: str) -> list[_Span]:
    """Split into paragraph spans, preserving offsets into the original text.

    Runs of the delimiter collapse (a document with three blank lines between
    paragraphs yields two paragraphs, not four empty ones), and surrounding
    whitespace is trimmed from each span so offsets point at real content.
    """
    spans: list[_Span] = []
    pattern = re.compile(f"(?:{re.escape(delimiter)})+")
    cursor = 0
    for match in pattern.finditer(text):
        spans.append(_Span(cursor, match.start()))
        cursor = match.end()
    spans.append(_Span(cursor, len(text)))

    trimmed: list[_Span] = []
    for span in spans:
        start, end = span.start, span.end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append(_Span(start, end))
    return trimmed


def _is_abbreviation(text: str, terminator_index: int) -> bool:
    """True if the period at `terminator_index` closes a known abbreviation."""
    if text[terminator_index] != ".":
        return False
    start = terminator_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start:terminator_index].lower().rstrip(".")
    return word in _ABBREVIATIONS


def _sentence_boundaries(text: str, span: _Span) -> list[int]:
    """Absolute offsets just past each sentence terminator within `span`."""
    segment = text[span.start : span.end]
    boundaries: list[int] = []
    for match in _SENTENCE_END.finditer(segment):
        local_terminator = match.start()
        if _is_abbreviation(segment, local_terminator):
            continue
        boundaries.append(span.start + match.end())
    return boundaries


def _split_oversized(text: str, span: _Span, max_chars: int) -> list[_Span]:
    """Split a span exceeding `max_chars`.

    Prefers the furthest sentence boundary at or below the limit (FR-ING-04). When no
    such boundary exists the span is hard-split at the limit (FR-ING-04a) — the caller
    is told via the returned `had_hard_split` signal in `chunk_text`.
    """
    if span.length <= max_chars:
        return [span]

    pieces: list[_Span] = []
    cursor = span.start
    while span.end - cursor > max_chars:
        limit = cursor + max_chars
        candidates = [b for b in _sentence_boundaries(text, _Span(cursor, limit)) if b > cursor]
        cut = candidates[-1] if candidates else limit
        pieces.append(_Span(cursor, cut))
        # Skip whitespace so the next piece starts on real content.
        while cut < span.end and text[cut].isspace():
            cut += 1
        cursor = cut
    if cursor < span.end:
        pieces.append(_Span(cursor, span.end))
    return pieces


def chunk_text(
    text: str,
    source_path: str,
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
    paragraph_delimiter: str = "\n\n",
) -> tuple[list[Chunk], list[str]]:
    """Chunk one document.

    Returns the chunks and a list of human-readable warnings (currently: hard splits,
    FR-ING-04a). Ordinals are contiguous from 0; offsets index into `text`.

    Order of operations matters and is specified: merge short paragraphs forward
    (FR-ING-03) *before* splitting oversized ones (FR-ING-04), so that a merge can never
    itself produce an over-limit chunk that escapes the split.
    """
    warnings: list[str] = []
    paragraphs = _paragraph_spans(text, paragraph_delimiter)
    if not paragraphs:
        return [], warnings

    # --- Pass 1: merge forward while below the minimum (FR-ING-03). ---
    merged: list[_Span] = []
    pending: _Span | None = None
    for span in paragraphs:
        pending = _Span(pending.start, span.end) if pending else span
        if pending.length >= min_chunk_chars:
            merged.append(pending)
            pending = None
    if pending is not None:
        # FR-ING-03a: a short trailing paragraph has nothing to merge into. It is kept
        # as its own chunk rather than discarded — dropping it would silently lose
        # content, which is a far worse failure than an undersized chunk.
        merged.append(pending)

    # --- Pass 2: split anything over the maximum (FR-ING-04/04a). ---
    final: list[_Span] = []
    for span in merged:
        if span.length <= max_chunk_chars:
            final.append(span)
            continue
        pieces = _split_oversized(text, span, max_chunk_chars)
        for piece in pieces:
            if piece.length > max_chunk_chars or not _sentence_boundaries(text, piece):
                if piece.length >= max_chunk_chars:
                    warnings.append(
                        f"hard split at max_chunk_chars in {source_path} "
                        f"(offset {piece.start}): no sentence boundary below the limit"
                    )
        final.extend(pieces)

    return (
        [
            Chunk(
                text=text[span.start : span.end],
                source_path=source_path,
                chunk_ordinal=ordinal,
                start_offset=span.start,
                end_offset=span.end,
            )
            for ordinal, span in enumerate(final)
        ],
        warnings,
    )


def discover_documents(source_dir: str | Path, supported_extensions: list[str]) -> tuple[list[Path], list[Path]]:
    """Find ingestible documents under `source_dir` (FR-ING-01/01a).

    Returns (supported, skipped). Skipped files are logged by the caller and never
    abort the run.
    """
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus source directory not found: {root}")
    allowed = {ext.lower() for ext in supported_extensions}
    supported: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        (supported if path.suffix.lower() in allowed else skipped).append(path)
    return supported, skipped
