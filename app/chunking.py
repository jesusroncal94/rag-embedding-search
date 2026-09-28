from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s)")

_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "vs",
        "cf",
        "approx",
        "no",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "inc",
        "ltd",
        "co",
        "st",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)


@dataclass(frozen=True)
class Chunk:
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
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _paragraph_spans(text: str, delimiter: str) -> list[_Span]:
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
    if text[terminator_index] != ".":
        return False
    start = terminator_index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    word = text[start:terminator_index].lower().rstrip(".")
    return word in _ABBREVIATIONS


def _sentence_boundaries(text: str, span: _Span) -> list[int]:
    segment = text[span.start : span.end]
    boundaries: list[int] = []
    for match in _SENTENCE_END.finditer(segment):
        local_terminator = match.start()
        if _is_abbreviation(segment, local_terminator):
            continue
        boundaries.append(span.start + match.end())
    return boundaries


def _split_oversized(text: str, span: _Span, max_chars: int) -> list[_Span]:
    if span.length <= max_chars:
        return [span]

    pieces: list[_Span] = []
    cursor = span.start
    while span.end - cursor > max_chars:
        limit = cursor + max_chars
        candidates = [b for b in _sentence_boundaries(text, _Span(cursor, limit)) if b > cursor]
        cut = candidates[-1] if candidates else limit
        pieces.append(_Span(cursor, cut))
        while cut < span.end and text[cut].isspace():
            cut += 1
        cursor = cut
    if cursor < span.end:
        pieces.append(_Span(cursor, span.end))
    return pieces


def _merge_short(paragraphs: list[_Span], min_chars: int) -> list[_Span]:
    merged: list[_Span] = []
    pending: _Span | None = None
    for span in paragraphs:
        pending = _Span(pending.start, span.end) if pending else span
        if pending.length >= min_chars:
            merged.append(pending)
            pending = None
    if pending is not None:
        merged.append(pending)
    return merged


def _is_hard_split(text: str, piece: _Span, max_chars: int) -> bool:
    return piece.length == max_chars and not _sentence_boundaries(text, piece)


def _split_long(
    text: str, spans: list[_Span], max_chars: int, source_path: str
) -> tuple[list[_Span], list[str]]:
    final: list[_Span] = []
    warnings: list[str] = []
    for span in spans:
        if span.length <= max_chars:
            final.append(span)
            continue
        pieces = _split_oversized(text, span, max_chars)
        for piece in pieces:
            if _is_hard_split(text, piece, max_chars):
                warnings.append(
                    f"hard split at max_chunk_chars in {source_path} "
                    f"(offset {piece.start}): no sentence boundary below the limit"
                )
        final.extend(pieces)
    return final, warnings


def chunk_text(
    text: str,
    source_path: str,
    *,
    min_chunk_chars: int,
    max_chunk_chars: int,
    paragraph_delimiter: str = "\n\n",
) -> tuple[list[Chunk], list[str]]:
    """Merges short paragraphs before splitting long ones, so a merge can never escape the limit."""
    paragraphs = _paragraph_spans(text, paragraph_delimiter)
    merged = _merge_short(paragraphs, min_chunk_chars)
    final, warnings = _split_long(text, merged, max_chunk_chars, source_path)
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


def discover_documents(
    source_dir: str | Path, supported_extensions: list[str]
) -> tuple[list[Path], list[Path]]:
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
