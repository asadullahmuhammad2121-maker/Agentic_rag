"""CSV document parser."""

from __future__ import annotations

import csv
from io import StringIO

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.services.ingestion.base import DocumentParser, ExtractedSection, NormalizedDocument

logger = get_logger(__name__)


class CsvParser(DocumentParser):
    """Parse CSV rows into one section per data row."""

    @property
    def file_type(self) -> str:
        return "csv"

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
                "CSV file is not valid UTF-8",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        if not text.strip():
            raise InvalidDocumentError(
                "CSV file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        try:
            reader = csv.DictReader(StringIO(text))
            if reader.fieldnames is None:
                raise InvalidDocumentError(
                    "CSV file is missing a header row",
                    details={"reason": "corrupted_file", "file_type": self.file_type},
                )
            rows = list(reader)
        except csv.Error as exc:
            raise InvalidDocumentError(
                "CSV file is corrupted or unreadable",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        sections: list[ExtractedSection] = []
        for row_index, row in enumerate(rows, start=1):
            parts = [f"{key}: {value}" for key, value in row.items() if value not in (None, "")]
            row_text = " | ".join(parts).strip()
            if not row_text:
                continue
            sections.append(
                ExtractedSection(
                    text=row_text,
                    section_index=len(sections),
                    page_number=1,
                    section=f"row:{row_index}",
                )
            )

        if not sections:
            raise InvalidDocumentError(
                "CSV file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        logger.info(
            "csv_parsed",
            extra={
                "operation": "parse_csv",
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
