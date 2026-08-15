"""Unit tests for PDF extraction."""

from __future__ import annotations

import pytest

from app.core.exceptions import InvalidDocumentError
from app.services.ingestion.pdf_extractor import PdfTextExtractor
from tests.helpers.pdf_fixtures import (
    build_corrupt_pdf_bytes,
    build_empty_pdf_bytes,
    build_pdf_bytes,
)


def test_extract_valid_pdf_preserves_page_numbers() -> None:
    content = build_pdf_bytes(["Page one text", "Page two text"])
    result = PdfTextExtractor().extract(content)

    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Page one text" in result.pages[0].text
    assert "Page two text" in result.pages[1].text


def test_extract_rejects_empty_file() -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        PdfTextExtractor().extract(build_empty_pdf_bytes())
    assert exc_info.value.details.get("reason") == "empty_file"


def test_extract_rejects_corrupted_pdf() -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        PdfTextExtractor().extract(build_corrupt_pdf_bytes())
    assert exc_info.value.details.get("reason") == "corrupted_pdf"


def test_extract_rejects_pdf_without_text() -> None:
    # Whitespace-only pages should be treated as empty extractable text.
    content = build_pdf_bytes(["   ", "\n"])
    with pytest.raises(InvalidDocumentError) as exc_info:
        PdfTextExtractor().extract(content)
    assert exc_info.value.details.get("reason") == "empty_text"
