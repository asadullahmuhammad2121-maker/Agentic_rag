"""Generic section heading detection and grouping for document navigation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PAGE_MARKER = re.compile(r"^Page\s+\d+\s*$", re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$", re.DOTALL)
_EMBEDDED_NUMBERED = re.compile(
    r"(?:^|\n|\s)\d+[.)]\s+([A-Za-z][\w\s\-]{1,80}?)(?:\s*$|\s*\n|\s+[A-Z])",
)
_STANDALONE_TITLE = re.compile(r"^([A-Z][A-Za-z0-9\s\-]{2,60})\s*$")
_LABEL_HEADING = re.compile(r"^\s*([A-Za-z][\w\s\-]{1,80}):\s*$")
_ORDERED_NUMBERED = re.compile(r"(?:^|\n)\s*\d+[.)]\s+([^\n:]+?)(?:\s*$|\s*\n|\s*:)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """Contiguous chunk range sharing one section heading."""

    label: str | None
    start_index: int
    end_index: int


def labels_match(left: str, right: str) -> bool:
    """True when two section labels refer to the same heading."""
    left_cf = left.casefold().strip()
    right_cf = right.casefold().strip()
    if not left_cf or not right_cf:
        return False
    return left_cf == right_cf or left_cf in right_cf or right_cf in left_cf


def extract_heading_label(text: str) -> str | None:
    """Return the primary section heading label in a chunk, if any."""
    stripped = text.strip()
    if not stripped:
        return None
    if _PAGE_MARKER.match(stripped):
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].isdigit():
        second = lines[1]
        label_match = re.match(r"^([A-Za-z][\w\s\-]{1,80})(?::|\s*$)", second)
        if label_match is not None:
            return label_match.group(1).strip()

    match = _NUMBERED_HEADING.match(stripped)
    if match is not None and len(stripped) < 120:
        return match.group(1).strip()

    match = _LABEL_HEADING.match(stripped)
    if match is not None:
        return match.group(1).strip()

    embedded: str | None = None
    for match in _EMBEDDED_NUMBERED.finditer(stripped):
        candidate = match.group(1).strip()
        if len(candidate) <= 60:
            embedded = candidate
    if embedded is not None:
        return embedded

    if len(stripped) < 80 and not stripped.endswith("."):
        match = _STANDALONE_TITLE.match(stripped)
        if match is not None and not _PAGE_MARKER.match(match.group(1)):
            words = match.group(1).split()
            if words and all(word[0].isupper() for word in words if word):
                return match.group(1).strip()

    return None


def extract_ordered_heading_labels(text: str) -> list[str]:
    """Return section labels found in text in document order."""
    labels: list[str] = []
    seen: set[str] = set()
    for match in _ORDERED_NUMBERED.finditer(text):
        label = match.group(1).strip()
        key = label.casefold()
        if key and key not in seen:
            seen.add(key)
            labels.append(label)
    return labels


def build_sections(chunks: list[tuple[int, str]]) -> list[DocumentSection]:
    """Group ordered (chunk_index, text) pairs into contiguous sections."""
    ordered = sorted(chunks, key=lambda item: item[0])
    if not ordered:
        return []

    sections: list[DocumentSection] = []
    index = 0
    while index < len(ordered):
        chunk_index, text = ordered[index]
        heading = extract_heading_label(text)
        if heading is None:
            start = chunk_index
            index += 1
            while index < len(ordered):
                next_heading = extract_heading_label(ordered[index][1])
                if next_heading is not None:
                    break
                index += 1
            end = ordered[index - 1][0]
            sections.append(
                DocumentSection(label=None, start_index=start, end_index=end),
            )
            continue

        label = heading
        start = chunk_index
        index += 1
        while index < len(ordered):
            next_heading = extract_heading_label(ordered[index][1])
            if next_heading is not None and not labels_match(next_heading, label):
                break
            index += 1
        end = ordered[index - 1][0]
        sections.append(
            DocumentSection(label=label, start_index=start, end_index=end),
        )

    return sections


def find_section_for_reference(
    sections: list[DocumentSection],
    *,
    anchor_index: int,
    reference: str,
) -> DocumentSection | None:
    """Locate the section that contains the reference label near the anchor."""
    for section in sections:
        if (
            section.start_index <= anchor_index <= section.end_index
            and section.label is not None
            and labels_match(section.label, reference)
        ):
            return section

    for section in sections:
        if section.label is not None and labels_match(section.label, reference):
            return section

    for section in sections:
        if section.start_index <= anchor_index <= section.end_index:
            return section

    return None


def adjacent_section(
    sections: list[DocumentSection],
    *,
    reference_section: DocumentSection,
    direction: str,
) -> DocumentSection | None:
    """Return the section immediately before/after the reference section."""
    for index, section in enumerate(sections):
        if section.start_index != reference_section.start_index:
            continue
        if direction == "after":
            if index + 1 < len(sections):
                return sections[index + 1]
            return None
        if index == 0:
            return None
        return sections[index - 1]
    return None


def successor_heading_present(combined_text: str, reference: str, *, direction: str) -> bool:
    """True when combined text already contains the next/previous section heading."""
    headings = extract_ordered_heading_labels(combined_text)
    if not headings:
        return False

    reference_index: int | None = None
    for index, heading in enumerate(headings):
        if labels_match(heading, reference):
            reference_index = index
            break
    if reference_index is None:
        return False

    if direction == "after":
        return reference_index + 1 < len(headings)
    return reference_index > 0
