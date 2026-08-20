"""Shared chunking helpers."""

from __future__ import annotations

import math
import re
from typing import Final

from app.services.chunking.base import TextChunk, TextSegment
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.structured_blocks import split_structured_text
from app.services.ingestion.base import ExtractedSection

SENTENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
)

RECURSIVE_SEPARATORS: Final[tuple[str, ...]] = (
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
)


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Build a deterministic chunk identifier."""
    return f"{document_id}:{chunk_index:05d}"


def section_to_segment(section: ExtractedSection) -> TextSegment | None:
    text = section.text.strip()
    if not text:
        return None
    origin = section.text.find(text)
    if origin < 0:
        origin = 0
    page_number = section.page_number if section.page_number is not None else 1
    return TextSegment(
        text=text,
        start_char=origin,
        end_char=origin + len(text),
        page_number=page_number,
        section=section.section,
    )


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_PATTERN.split(text.strip()) if part.strip()]
    return parts or [text.strip()]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def recursive_split_text(
    text: str,
    *,
    config: ChunkingConfig,
    separators: tuple[str, ...] = RECURSIVE_SEPARATORS,
) -> list[tuple[str, int, int]]:
    """Split text using progressively finer separators until size fits."""
    cleaned = text.strip()
    if not cleaned:
        return []

    origin = text.find(cleaned)
    if origin < 0:
        origin = 0
        working = cleaned
    else:
        working = text[origin : origin + len(cleaned)]

    if len(working) <= config.chunk_size:
        return [(working, origin, origin + len(working))]

    return _split_recursive(working, origin=origin, config=config, separators=separators)


def _split_recursive(
    text: str,
    *,
    origin: int,
    config: ChunkingConfig,
    separators: tuple[str, ...],
) -> list[tuple[str, int, int]]:
    if len(text) <= config.chunk_size:
        stripped = text.strip()
        if not stripped:
            return []
        leading = len(text) - len(text.lstrip())
        start = origin + leading
        return [(stripped, start, start + len(stripped))]

    if not separators:
        return _split_by_length(text, origin=origin, config=config)

    separator = separators[0]
    remaining = separators[1:]
    if separator not in text:
        return _split_recursive(text, origin=origin, config=config, separators=remaining)

    parts = text.split(separator)
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for index, part in enumerate(parts):
        piece = part if index == len(parts) - 1 else part + separator
        if not piece:
            cursor += len(piece)
            continue
        piece_origin = origin + cursor
        if len(piece.strip()) <= config.chunk_size:
            stripped = piece.strip()
            if stripped:
                leading = len(piece) - len(piece.lstrip())
                start = piece_origin + leading
                segments.append((stripped, start, start + len(stripped)))
        else:
            segments.extend(
                _split_recursive(
                    piece,
                    origin=piece_origin,
                    config=config,
                    separators=remaining,
                )
            )
        cursor += len(piece)
    return segments


def _split_by_length(
    text: str,
    *,
    origin: int,
    config: ChunkingConfig,
) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + config.chunk_size, length)
        piece = text[start:end].strip()
        if piece:
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            abs_start = origin + start + leading
            segments.append((piece, abs_start, abs_start + len(piece)))
        if end >= length:
            break
        next_start = end - config.chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return segments


def fixed_split_text(text: str, *, config: ChunkingConfig) -> list[tuple[str, int, int]]:
    """Character-window splitting with optional word-boundary adjustment."""
    cleaned = text.strip()
    if not cleaned:
        return []

    origin = text.find(cleaned)
    if origin < 0:
        origin = 0
        working = cleaned
    else:
        working = text[origin : origin + len(cleaned)]

    if len(working) <= config.chunk_size:
        return [(working, origin, origin + len(working))]

    parts: list[tuple[str, int, int]] = []
    start = 0
    length = len(working)
    while start < length:
        end = min(start + config.chunk_size, length)
        if end < length:
            break_at = working.rfind(" ", start, end)
            if break_at > start:
                end = break_at

        piece = working[start:end]
        stripped = piece.strip()
        if stripped:
            leading = len(piece) - len(piece.lstrip())
            abs_start = origin + start + leading
            abs_end = abs_start + len(stripped)
            parts.append((stripped, abs_start, abs_end))

        if end >= length:
            break

        next_start = end - config.chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return parts


def split_text_with_structured_blocks(
    text: str,
    *,
    config: ChunkingConfig,
    preserve_prose_intact: bool,
) -> list[tuple[str, int, int]]:
    """
    Split text while preserving structured heading/list blocks when possible.

    Structured list blocks are kept intact when they fit within ``max_chunk_size``.
    Prose blocks are kept intact only when ``preserve_prose_intact`` is True
    (structure strategy). Otherwise prose falls back to recursive splitting.
    """
    pieces: list[tuple[str, int, int]] = []
    for block_text, start, end, is_list_block in split_structured_text(text):
        keep_intact = len(block_text) <= config.max_chunk_size and (
            is_list_block or preserve_prose_intact
        )
        if keep_intact:
            pieces.append((block_text, start, end))
            continue

        block_origin = text.find(block_text, max(0, start - 1))
        if block_origin < 0:
            block_origin = start
        for piece_text, piece_start, piece_end in recursive_split_text(
            block_text,
            config=config,
        ):
            pieces.append(
                (
                    piece_text,
                    block_origin + piece_start,
                    block_origin + piece_end,
                )
            )
    return pieces


def enforce_size_bounds(
    segments: list[TextSegment],
    *,
    config: ChunkingConfig,
) -> list[TextSegment]:
    """Merge tiny segments and split oversized ones."""
    if not segments:
        return []

    bounded: list[TextSegment] = []
    for segment in segments:
        if len(segment.text) > config.max_chunk_size:
            for text, start, end in fixed_split_text(
                segment.text,
                config=ChunkingConfig(
                    strategy=config.strategy,
                    chunk_size=config.max_chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    min_chunk_size=config.min_chunk_size,
                    max_chunk_size=config.max_chunk_size,
                    semantic_similarity_threshold=config.semantic_similarity_threshold,
                ),
            ):
                bounded.append(
                    TextSegment(
                        text=text,
                        start_char=segment.start_char + start,
                        end_char=segment.start_char + end,
                        page_number=segment.page_number,
                        section=segment.section,
                    )
                )
            continue
        bounded.append(segment)

    merged: list[TextSegment] = []
    for segment in bounded:
        if (
            merged
            and len(segment.text) < config.min_chunk_size
            and merged[-1].page_number == segment.page_number
            and merged[-1].section == segment.section
        ):
            combined_text = f"{merged[-1].text} {segment.text}".strip()
            if len(combined_text) <= config.chunk_size:
                merged[-1] = TextSegment(
                    text=combined_text,
                    start_char=merged[-1].start_char,
                    end_char=segment.end_char,
                    page_number=merged[-1].page_number,
                    section=merged[-1].section,
                )
                continue
        merged.append(segment)

    return merged


def segments_to_chunks(
    segments: list[TextSegment],
    *,
    document_id: str,
    filename: str,
    file_type: str,
    source: str,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for index, segment in enumerate(segments):
        chunks.append(
            TextChunk(
                text=segment.text,
                chunk_id=make_chunk_id(document_id, index),
                chunk_index=index,
                page_number=segment.page_number,
                document_id=document_id,
                filename=filename,
                start_char=segment.start_char,
                end_char=segment.end_char,
                file_type=file_type,
                section=segment.section,
                source=source,
            )
        )
    return chunks
