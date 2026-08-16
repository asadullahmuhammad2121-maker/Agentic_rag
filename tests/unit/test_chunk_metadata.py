"""Unit tests for chunk payload metadata consistency."""

from __future__ import annotations

from app.services.ingestion.metadata import REQUIRED_CHUNK_METADATA_FIELDS, build_chunk_payload


def test_build_chunk_payload_includes_required_fields() -> None:
    payload = build_chunk_payload(
        document_id="doc-1",
        filename="notes.txt",
        file_type="txt",
        source="notes.txt",
        page_number=1,
        section="Introduction",
        chunk_id="doc-1:00000",
        chunk_index=0,
        chunking_strategy="fixed",
        extra={"text": "hello"},
    )
    for field_name in REQUIRED_CHUNK_METADATA_FIELDS:
        assert field_name in payload
    assert payload["section"] == "Introduction"
    assert payload["chunking_strategy"] == "fixed"


def test_build_chunk_payload_allows_null_section() -> None:
    payload = build_chunk_payload(
        document_id="doc-1",
        filename="notes.txt",
        file_type="txt",
        source="notes.txt",
        page_number=1,
        section=None,
        chunk_id="doc-1:00000",
        chunk_index=0,
        chunking_strategy="fixed",
        extra={"text": "hello"},
    )
    assert payload["section"] is None
