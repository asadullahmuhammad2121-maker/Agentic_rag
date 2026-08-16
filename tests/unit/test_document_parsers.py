"""Unit tests for format-specific document parsers."""

from __future__ import annotations

import pytest

from app.core.exceptions import InvalidDocumentError
from app.services.ingestion.parsers.csv import CsvParser
from app.services.ingestion.parsers.docx import DocxParser
from app.services.ingestion.parsers.json import JsonParser
from app.services.ingestion.parsers.markdown import MarkdownParser
from app.services.ingestion.parsers.pdf import PdfParser
from app.services.ingestion.parsers.txt import TxtParser
from tests.helpers.document_fixtures import (
    build_corrupt_csv_bytes,
    build_corrupt_docx_bytes,
    build_corrupt_json_bytes,
    build_csv_bytes,
    build_docx_bytes,
    build_json_bytes,
    build_markdown_bytes,
    build_txt_bytes,
)
from tests.helpers.pdf_fixtures import build_corrupt_pdf_bytes, build_pdf_bytes


def test_pdf_parser_preserves_page_numbers() -> None:
    content = build_pdf_bytes(["Page one text", "Page two text"])
    result = PdfParser().parse(content, filename="sample.pdf")

    assert result.file_type == "pdf"
    assert result.page_count == 2
    assert len(result.sections) == 2
    assert result.sections[0].page_number == 1
    assert "Page one text" in result.sections[0].text


def test_pdf_parser_rejects_corrupted_file() -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        PdfParser().parse(build_corrupt_pdf_bytes(), filename="bad.pdf")
    assert exc_info.value.details.get("reason") == "corrupted_pdf"


def test_txt_parser_extracts_content() -> None:
    result = TxtParser().parse(build_txt_bytes("Hello plain text"), filename="notes.txt")
    assert result.file_type == "txt"
    assert result.section_count == 1
    assert result.sections[0].text == "Hello plain text"


def test_markdown_parser_splits_by_headings() -> None:
    content = build_markdown_bytes("# Title\nBody one\n\n## Section\nBody two")
    result = MarkdownParser().parse(content, filename="readme.md")
    assert result.file_type == "markdown"
    assert result.section_count == 2
    assert result.sections[0].section == "Title"
    assert "Body one" in result.sections[0].text
    assert result.sections[1].section == "Section"


def test_csv_parser_extracts_rows() -> None:
    content = build_csv_bytes(
        [
            {"name": "Alice", "role": "Engineer"},
            {"name": "Bob", "role": "Designer"},
        ]
    )
    result = CsvParser().parse(content, filename="people.csv")
    assert result.file_type == "csv"
    assert result.section_count == 2
    assert result.sections[0].section == "row:1"
    assert "Alice" in result.sections[0].text
    assert "Bob" in result.sections[1].text


def test_json_parser_extracts_scalar_paths() -> None:
    content = build_json_bytes({"title": "Report", "author": "Ada", "tags": ["rag", "search"]})
    result = JsonParser().parse(content, filename="data.json")
    assert result.file_type == "json"
    assert result.section_count == 4
    joined = " ".join(section.text for section in result.sections)
    assert "$.title: Report" in joined
    assert "$.author: Ada" in joined


def test_docx_parser_extracts_paragraphs() -> None:
    content = build_docx_bytes(
        [
            ("Introduction", "Heading 1"),
            ("This is the intro paragraph.", None),
            ("Details", "Heading 2"),
            ("More detail text.", None),
        ]
    )
    result = DocxParser().parse(content, filename="report.docx")
    assert result.file_type == "docx"
    assert result.section_count >= 2
    assert any("intro paragraph" in section.text for section in result.sections)


def test_unsupported_parser_registry_rejected() -> None:
    from app.services.ingestion.parser_registry import DocumentParserRegistry

    registry = DocumentParserRegistry()
    with pytest.raises(InvalidDocumentError) as exc_info:
        registry.get_parser("exe")
    assert exc_info.value.details.get("reason") == "unsupported_file_type"


@pytest.mark.parametrize(
    ("parser", "content", "filename"),
    [
        (TxtParser(), b"", "empty.txt"),
        (MarkdownParser(), b"   ", "blank.md"),
        (JsonParser(), build_corrupt_json_bytes(), "bad.json"),
        (CsvParser(), build_corrupt_csv_bytes(), "bad.csv"),
        (DocxParser(), build_corrupt_docx_bytes(), "bad.docx"),
    ],
)
def test_parsers_reject_corrupted_or_empty_files(
    parser: object,
    content: bytes,
    filename: str,
) -> None:
    with pytest.raises(InvalidDocumentError):
        parser.parse(content, filename=filename)  # type: ignore[attr-defined]
