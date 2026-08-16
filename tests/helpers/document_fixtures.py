"""Helpers for building non-PDF document fixtures in tests."""

from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO

from docx import Document as DocxDocument


def build_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def build_markdown_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def build_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    if not rows:
        return b""
    fieldnames = list(rows[0].keys())
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def build_docx_bytes(paragraphs: list[tuple[str, str | None]]) -> bytes:
    """
    Build a DOCX file.

    Each entry is ``(text, style_name_or_none)`` where style can be ``Heading 1`` etc.
    """
    document = DocxDocument()
    for text, style in paragraphs:
        paragraph = document.add_paragraph(text)
        if style:
            paragraph.style = style
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def build_corrupt_docx_bytes() -> bytes:
    return b"PK\x03\x04this-is-not-a-valid-docx"


def build_corrupt_json_bytes() -> bytes:
    return b"{not valid json"


def build_corrupt_csv_bytes() -> bytes:
    return b"\xff\xfe invalid csv bytes"
