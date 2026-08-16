"""Retrieval data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A retrieved chunk with citation-ready metadata."""

    chunk_id: str
    text: str
    document_id: str
    filename: str
    file_type: str
    source: str
    page_number: int
    section: str | None
    chunk_index: int
    chunking_strategy: str
    score: float
