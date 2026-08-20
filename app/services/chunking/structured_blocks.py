"""Detect and preserve structured document blocks during chunking."""

from __future__ import annotations

import re
from typing import TypeAlias

StructuredTextBlock: TypeAlias = tuple[str, int, int, bool]

_NUMBERED_LIST_LINE = re.compile(r"^\d+[.)]\s+\S")
_BULLET_LIST_LINE = re.compile(r"^[-*•]\s+\S")
_LABEL_LIST_LINE = re.compile(r"^[A-Za-z][\w\s\-]{0,60}:\s+\S")
_TRAILING_HEADING = re.compile(
    r"^(?P<prose>.+[.!?])\s+(?P<heading>.{1,100})$"
)


def is_list_line(line: str) -> bool:
    """Return True when a line looks like a numbered, bullet, or label list item."""
    stripped = line.strip()
    if not stripped:
        return False
    return (
        _NUMBERED_LIST_LINE.match(stripped) is not None
        or _BULLET_LIST_LINE.match(stripped) is not None
        or _LABEL_LIST_LINE.match(stripped) is not None
    )


def extract_trailing_heading(line: str) -> tuple[str, str | None]:
    """
    Split a prose line into body and a trailing heading suffix.

    Handles PDF-style extraction where a heading sits at the end of a paragraph
    line immediately before a list (for example ``...answers. Section Title 1``).
    """
    stripped = line.strip()
    match = _TRAILING_HEADING.match(stripped)
    if match is None:
        return stripped, None
    heading = match.group("heading").strip()
    prose = match.group("prose").strip()
    if not heading or is_list_line(heading):
        return stripped, None
    return prose, heading


def split_structured_text(text: str) -> list[StructuredTextBlock]:
    """
    Split text into logical blocks, keeping headings with following list runs.

    A list run requires at least two consecutive list-item lines. Each block
    includes a flag indicating whether it is a heading/list structured block.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    line_spans = _line_spans(text)
    if not line_spans:
        origin = text.find(cleaned)
        if origin < 0:
            origin = 0
        return [(cleaned, origin, origin + len(cleaned), False)]

    structured_ranges = _structured_list_ranges(text, line_spans)
    if not structured_ranges:
        origin = text.find(cleaned)
        if origin < 0:
            origin = 0
        return [(cleaned, origin, origin + len(cleaned), False)]

    structured_ranges.sort(key=lambda item: item[0])
    blocks: list[StructuredTextBlock] = []
    cursor = 0

    for start, end in structured_ranges:
        if cursor < start:
            blocks.extend(
                _split_prose(text[cursor:start], origin=cursor, is_list_block=False)
            )
        block_text = text[start:end].strip()
        if block_text:
            abs_start = text.find(block_text, start)
            if abs_start < 0:
                abs_start = start
            blocks.append((block_text, abs_start, abs_start + len(block_text), True))
        cursor = max(cursor, end)

    if cursor < len(text):
        blocks.extend(_split_prose(text[cursor:], origin=cursor, is_list_block=False))

    fallback_origin = text.find(cleaned)
    if fallback_origin < 0:
        fallback_origin = 0
    return blocks or [(cleaned, fallback_origin, fallback_origin + len(cleaned), False)]


def _split_prose(
    prose: str,
    *,
    origin: int,
    is_list_block: bool,
) -> list[StructuredTextBlock]:
    """Split remaining prose on paragraph boundaries."""
    stripped = prose.strip()
    if not stripped:
        return []

    blocks: list[StructuredTextBlock] = []
    search_from = 0
    for part in re.split(r"\n\s*\n", stripped):
        piece = part.strip()
        if not piece:
            continue
        local = prose.find(piece, search_from)
        if local < 0:
            local = prose.find(piece)
        if local < 0:
            continue
        abs_start = origin + local
        blocks.append((piece, abs_start, abs_start + len(piece), is_list_block))
        search_from = local + len(piece)
    return blocks


def _line_spans(text: str) -> list[tuple[str, int, int]]:
    """Return (stripped_line, start, end) offsets for each non-empty line."""
    spans: list[tuple[str, int, int]] = []
    offset = 0
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            leading = line.find(stripped)
            start = offset + leading
            spans.append((stripped, start, start + len(stripped)))
        offset += len(line)
        if index < len(lines) - 1:
            offset += 1
    return spans


def _list_run_length(spans: list[tuple[str, int, int]], start: int) -> int:
    length = 0
    index = start
    while index < len(spans) and is_list_line(spans[index][0]):
        length += 1
        index += 1
    return length


def _blank_line_before(text: str, spans: list[tuple[str, int, int]], line_index: int) -> bool:
    if line_index <= 0:
        return True
    _, _, previous_end = spans[line_index - 1]
    _, current_start, _ = spans[line_index]
    gap = text[previous_end:current_start]
    return "\n\n" in gap or gap.strip() == ""


def _structured_list_ranges(
    text: str,
    spans: list[tuple[str, int, int]],
) -> list[tuple[int, int]]:
    """Find character ranges for heading + list blocks."""
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(spans):
        run_length = _list_run_length(spans, index)
        if run_length < 2:
            index += 1
            continue

        list_start = index
        list_end = index + run_length
        char_start = spans[list_start][1]
        char_end = spans[list_end - 1][2]

        heading_index = list_start - 1
        if heading_index >= 0:
            previous_line, previous_start, _previous_end = spans[heading_index]
            if not is_list_line(previous_line):
                if _blank_line_before(text, spans, heading_index):
                    char_start = previous_start
                else:
                    prose, trailing = extract_trailing_heading(previous_line)
                    if trailing:
                        char_start = _previous_end - len(trailing)
                    elif len(previous_line) <= 100 and not previous_line.endswith((".", "!", "?")):
                        char_start = previous_start

        ranges.append((char_start, char_end))
        index = list_end
    return _merge_ranges(ranges)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
