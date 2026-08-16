"""Document ingestion services."""

from app.services.ingestion.base import (
    DocumentParser,
    ExtractedDocument,
    ExtractedPage,
    ExtractedSection,
    NormalizedDocument,
)
from app.services.ingestion.format_detection import detect_file_type
from app.services.ingestion.parser_registry import DocumentParserRegistry
from app.services.ingestion.parsers.pdf import PdfParser, PdfTextExtractor
from app.services.ingestion.service import DocumentIngestionService, IngestedDocument

__all__ = [
    "DocumentIngestionService",
    "DocumentParser",
    "DocumentParserRegistry",
    "ExtractedDocument",
    "ExtractedPage",
    "ExtractedSection",
    "IngestedDocument",
    "NormalizedDocument",
    "PdfParser",
    "PdfTextExtractor",
    "detect_file_type",
]
