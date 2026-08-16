"""JSON document parser."""

from __future__ import annotations

import json
from typing import Any

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.services.ingestion.base import DocumentParser, ExtractedSection, NormalizedDocument

logger = get_logger(__name__)


class JsonParser(DocumentParser):
    """Parse JSON into readable text sections keyed by top-level paths."""

    @property
    def file_type(self) -> str:
        return "json"

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
                "JSON file is not valid UTF-8",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        if not text.strip():
            raise InvalidDocumentError(
                "JSON file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidDocumentError(
                "JSON file is corrupted or unreadable",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        sections = self._collect_sections(payload, path="$")
        if not sections:
            raise InvalidDocumentError(
                "JSON file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        logger.info(
            "json_parsed",
            extra={
                "operation": "parse_json",
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

    def _collect_sections(self, value: Any, *, path: str) -> list[ExtractedSection]:
        sections: list[ExtractedSection] = []
        self._walk_value(value, path=path, sections=sections)
        for index, section in enumerate(sections):
            sections[index] = ExtractedSection(
                text=section.text,
                section_index=index,
                page_number=section.page_number,
                section=section.section,
            )
        return sections

    def _walk_value(self, value: Any, *, path: str, sections: list[ExtractedSection]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                self._walk_value(item, path=f"{path}.{key}", sections=sections)
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                self._walk_value(item, path=f"{path}[{index}]", sections=sections)
            return

        rendered = self._render_scalar(value)
        if not rendered:
            return
        sections.append(
            ExtractedSection(
                text=f"{path}: {rendered}",
                section_index=len(sections),
                page_number=1,
                section=path,
            )
        )

    @staticmethod
    def _render_scalar(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()
