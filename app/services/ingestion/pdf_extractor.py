"""PDF text extraction with page-number preservation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ExtractedPage:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass(slots=True, frozen=True)
class ExtractedDocument:
    """Full PDF extraction result."""

    pages: list[ExtractedPage]
    page_count: int


class PdfTextExtractor:
    """Extract plain text from PDF bytes while preserving page numbers."""

    def extract(self, content: bytes) -> ExtractedDocument:
        if not content:
            raise InvalidDocumentError(
                "PDF file is empty",
                details={"reason": "empty_file"},
            )

        try:
            reader = PdfReader(BytesIO(content), strict=False)
        except PdfReadError as exc:
            logger.warning(
                "pdf_read_failed",
                extra={"operation": "extract_pdf", "error_type": type(exc).__name__},
            )
            raise InvalidDocumentError(
                "PDF file is corrupted or unreadable",
                details={"reason": "corrupted_pdf"},
            ) from exc
        except Exception as exc:
            logger.warning(
                "pdf_parse_failed",
                extra={"operation": "extract_pdf", "error_type": type(exc).__name__},
            )
            raise InvalidDocumentError(
                "PDF file is corrupted or unreadable",
                details={"reason": "corrupted_pdf"},
            ) from exc

        if getattr(reader, "is_encrypted", False):
            # Attempt empty-password decrypt; fail closed if still encrypted.
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise InvalidDocumentError(
                    "Encrypted PDFs are not supported",
                    details={"reason": "encrypted_pdf"},
                ) from exc
            if decrypted == 0:
                raise InvalidDocumentError(
                    "Encrypted PDFs are not supported",
                    details={"reason": "encrypted_pdf"},
                )

        page_count = len(reader.pages)
        if page_count == 0:
            raise InvalidDocumentError(
                "PDF contains no pages",
                details={"reason": "no_pages"},
            )

        pages: list[ExtractedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "pdf_page_extract_failed",
                    extra={
                        "operation": "extract_pdf",
                        "page_number": index,
                        "error_type": type(exc).__name__,
                    },
                )
                raise InvalidDocumentError(
                    "Failed to extract text from PDF page",
                    details={"reason": "page_extract_failed", "page_number": index},
                ) from exc
            pages.append(ExtractedPage(page_number=index, text=raw_text.strip()))

        if all(not page.text for page in pages):
            raise InvalidDocumentError(
                "PDF contains no extractable text",
                details={"reason": "empty_text", "page_count": page_count},
            )

        logger.info(
            "pdf_extracted",
            extra={
                "operation": "extract_pdf",
                "page_count": page_count,
                "pages_with_text": sum(1 for page in pages if page.text),
            },
        )
        return ExtractedDocument(pages=pages, page_count=page_count)
