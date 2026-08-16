"""Plain text document parser."""

from __future__ import annotations

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.services.ingestion.base import DocumentParser, ExtractedSection, NormalizedDocument

logger = get_logger(__name__)


class TxtParser(DocumentParser):
    """Parse UTF-8 plain text files."""

    @property
    def file_type(self) -> str:
        return "txt"

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
                "Text file is not valid UTF-8",
                details={"reason": "corrupted_file", "file_type": self.file_type},
            ) from exc

        cleaned = text.strip()
        if not cleaned:
            raise InvalidDocumentError(
                "Text file contains no extractable content",
                details={"reason": "empty_text", "file_type": self.file_type},
            )

        section = ExtractedSection(
            text=cleaned,
            section_index=0,
            page_number=1,
            section=None,
        )
        logger.info(
            "txt_parsed",
            extra={"operation": "parse_txt", "source": filename, "text_length": len(cleaned)},
        )
        return NormalizedDocument(
            sections=[section],
            file_type=self.file_type,
            source=filename,
            page_count=1,
            section_count=1,
        )
