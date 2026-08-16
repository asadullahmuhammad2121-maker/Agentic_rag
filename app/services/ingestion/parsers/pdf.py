"""PDF document parser."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.services.ingestion.base import (
    DocumentParser,
    ExtractedDocument,
    ExtractedPage,
    ExtractedSection,
    NormalizedDocument,
)

logger = get_logger(__name__)


class PdfParser(DocumentParser):
    """Extract plain text from PDF bytes while preserving page numbers."""

    @property
    def file_type(self) -> str:
        return "pdf"

    def parse(self, content: bytes, *, filename: str) -> NormalizedDocument:
        extracted = self._extract_pages(content)
        sections = [
            ExtractedSection(
                text=page.text,
                section_index=index,
                page_number=page.page_number,
                section=None,
            )
            for index, page in enumerate(extracted.pages)
            if page.text
        ]
        return NormalizedDocument(
            sections=sections,
            file_type=self.file_type,
            source=filename,
            page_count=extracted.page_count,
            section_count=len(sections),
        )

    def extract(self, content: bytes) -> ExtractedDocument:
        """Backward-compatible PDF extraction API."""
        return self._extract_pages(content)

    def _extract_pages(self, content: bytes) -> ExtractedDocument:
        if not content:
            raise InvalidDocumentError(
                "Document file is empty",
                details={"reason": "empty_file", "file_type": self.file_type},
            )

        try:
            reader = PdfReader(BytesIO(content), strict=False)
        except PdfReadError as exc:
            logger.warning(
                "pdf_read_failed",
                extra={"operation": "parse_pdf", "error_type": type(exc).__name__},
            )
            raise InvalidDocumentError(
                "PDF file is corrupted or unreadable",
                details={"reason": "corrupted_pdf", "file_type": self.file_type},
            ) from exc
        except Exception as exc:
            logger.warning(
                "pdf_parse_failed",
                extra={"operation": "parse_pdf", "error_type": type(exc).__name__},
            )
            raise InvalidDocumentError(
                "PDF file is corrupted or unreadable",
                details={"reason": "corrupted_pdf", "file_type": self.file_type},
            ) from exc

        if getattr(reader, "is_encrypted", False):
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise InvalidDocumentError(
                    "Encrypted PDFs are not supported",
                    details={"reason": "encrypted_pdf", "file_type": self.file_type},
                ) from exc
            if decrypted == 0:
                raise InvalidDocumentError(
                    "Encrypted PDFs are not supported",
                    details={"reason": "encrypted_pdf", "file_type": self.file_type},
                )

        page_count = len(reader.pages)
        if page_count == 0:
            raise InvalidDocumentError(
                "PDF contains no pages",
                details={"reason": "no_pages", "file_type": self.file_type},
            )

        pages: list[ExtractedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "pdf_page_extract_failed",
                    extra={
                        "operation": "parse_pdf",
                        "page_number": index,
                        "error_type": type(exc).__name__,
                    },
                )
                raise InvalidDocumentError(
                    "Failed to extract text from PDF page",
                    details={
                        "reason": "page_extract_failed",
                        "file_type": self.file_type,
                        "page_number": index,
                    },
                ) from exc
            pages.append(ExtractedPage(page_number=index, text=raw_text.strip()))

        if all(not page.text for page in pages):
            raise InvalidDocumentError(
                "PDF contains no extractable text",
                details={"reason": "empty_text", "file_type": self.file_type, "page_count": page_count},
            )

        logger.info(
            "pdf_parsed",
            extra={
                "operation": "parse_pdf",
                "page_count": page_count,
                "pages_with_text": sum(1 for page in pages if page.text),
            },
        )
        return ExtractedDocument(pages=pages, page_count=page_count)


class PdfTextExtractor(PdfParser):
    """Backward-compatible alias for existing imports."""
