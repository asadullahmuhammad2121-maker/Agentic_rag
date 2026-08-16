"""PDF text extraction with page-number preservation.

Backward-compatible re-exports. Prefer ``app.services.ingestion.parsers.pdf``.
"""

from app.services.ingestion.base import ExtractedDocument, ExtractedPage
from app.services.ingestion.parsers.pdf import PdfParser, PdfTextExtractor

__all__ = ["ExtractedDocument", "ExtractedPage", "PdfParser", "PdfTextExtractor"]
