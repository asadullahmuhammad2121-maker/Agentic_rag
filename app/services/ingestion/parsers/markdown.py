"""Markdown document parser."""

from __future__ import annotations

import re

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.services.ingestion.base import DocumentParser, ExtractedSection, NormalizedDocument

logger = get_logger(__name__)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownParser(DocumentParser):
    """Parse Markdown files into heading-based sections."""

    @property
    def file_type(self) -> str:
        return "markdown"

    def parse(self, content: bytes, *, filename: str) -> NormalizedDocument:
        if not content:
            raise InvalidDocumentError(
                "Document file is empty",
                details={"reason": "empty_file", "file_type": self.file_type},
            )

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                "Markdown file is not valid UTF-8",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        cleaned = text.strip()
        if not cleaned:
            raise InvalidDocumentError(
                "Markdown file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        sections = self._split_by_headings(cleaned)
        logger.info(
            "markdown_parsed",
            extra={
                "operation": "parse_markdown",
                "source": filename,
                "section_count": len(sections),
            },
        )
        return NormalizedDocument(
            sections=sections,
            file_type=self.file_type,
            source=filename,
            page_count=1,
            section_count=len(sections),
        )

    def _split_by_headings(self, text: str) -> list[ExtractedSection]:
        matches = list(_HEADING_PATTERN.finditer(text))
        if not matches:
            return [
                ExtractedSection(
                    text=text,
                    section_index=0,
                    page_number=1,
                    section=None,
                )
            ]

        sections: list[ExtractedSection] = []
        for index, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            sections.append(
                ExtractedSection(
                    text=body,
                    section_index=len(sections),
                    page_number=1,
                    section=heading,
                )
            )

        if not sections:
            raise InvalidDocumentError(
                "Markdown file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )
        return sections
