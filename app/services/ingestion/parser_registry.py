"""Registry for format-specific document parsers."""

from __future__ import annotations

from app.core.exceptions import InvalidDocumentError
from app.services.ingestion.base import SUPPORTED_FILE_TYPES, DocumentParser
from app.services.ingestion.parsers.csv import CsvParser
from app.services.ingestion.parsers.docx import DocxParser
from app.services.ingestion.parsers.json import JsonParser
from app.services.ingestion.parsers.markdown import MarkdownParser
from app.services.ingestion.parsers.pdf import PdfParser
from app.services.ingestion.parsers.txt import TxtParser


class DocumentParserRegistry:
    """Resolve the parser implementation for a detected file type."""

    def __init__(self, parsers: dict[str, DocumentParser] | None = None) -> None:
        self._parsers = parsers or {
            "pdf": PdfParser(),
            "docx": DocxParser(),
            "txt": TxtParser(),
            "markdown": MarkdownParser(),
            "csv": CsvParser(),
            "json": JsonParser(),
        }

    def get_parser(self, file_type: str) -> DocumentParser:
        if file_type not in SUPPORTED_FILE_TYPES:
            raise InvalidDocumentError(
                "Unsupported file type",
                details={"reason": "unsupported_file_type", "file_type": file_type},
            )
        parser = self._parsers.get(file_type)
        if parser is None:
            raise InvalidDocumentError(
                "Unsupported file type",
                details={"reason": "unsupported_file_type", "file_type": file_type},
            )
        return parser
