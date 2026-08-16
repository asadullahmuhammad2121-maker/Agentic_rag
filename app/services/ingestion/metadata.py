"""Standard chunk metadata helpers for Qdrant payloads."""

from __future__ import annotations

from typing import Any

REQUIRED_CHUNK_METADATA_FIELDS: tuple[str, ...] = (
    "document_id",
    "filename",
    "file_type",
    "source",
    "page_number",
    "section",
    "chunk_id",
    "chunk_index",
    "chunking_strategy",
)


def build_chunk_payload(
    *,
    document_id: str,
    filename: str,
    file_type: str,
    source: str,
    page_number: int,
    section: str | None,
    chunk_id: str,
    chunk_index: int,
    chunking_strategy: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistent Qdrant payload for an ingested chunk."""
    payload = {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "source": source,
        "page_number": page_number,
        "section": section,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunking_strategy": chunking_strategy,
        **extra,
    }
    for field_name in REQUIRED_CHUNK_METADATA_FIELDS:
        if field_name not in payload:
            msg = f"Missing required chunk metadata field: {field_name}"
            raise ValueError(msg)
    return payload
