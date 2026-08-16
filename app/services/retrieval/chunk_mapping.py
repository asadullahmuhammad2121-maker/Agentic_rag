"""Map vector-store payloads to retrieved chunk models."""

from __future__ import annotations

from typing import Any

from app.services.retrieval.models import RetrievedChunk


def payload_to_retrieved_chunk(
    point_id: str,
    score: float,
    payload: dict[str, Any],
) -> RetrievedChunk:
    """Build a ``RetrievedChunk`` from stored chunk metadata."""
    page_number = payload.get("page_number", 0)
    chunk_index = payload.get("chunk_index", 0)
    try:
        page_number_int = int(page_number)
    except (TypeError, ValueError):
        page_number_int = 0
    try:
        chunk_index_int = int(chunk_index)
    except (TypeError, ValueError):
        chunk_index_int = 0

    section = payload.get("section")
    section_value = str(section) if section not in (None, "") else None
    chunk_id = payload.get("chunk_id")
    resolved_chunk_id = str(chunk_id) if chunk_id not in (None, "") else str(point_id)

    return RetrievedChunk(
        chunk_id=resolved_chunk_id,
        text=str(payload.get("text", "")),
        document_id=str(payload.get("document_id", "")),
        filename=str(payload.get("filename", "")),
        file_type=str(payload.get("file_type", payload.get("content_type", "unknown"))),
        source=str(payload.get("source", payload.get("filename", ""))),
        page_number=page_number_int,
        section=section_value,
        chunk_index=chunk_index_int,
        chunking_strategy=str(payload.get("chunking_strategy", "unknown")),
        score=float(score),
    )
