"""Unit tests for ChunkingService."""

from __future__ import annotations

import pytest

from app.services.chunking.service import ChunkingService
from app.services.ingestion.pdf_extractor import ExtractedPage


def test_chunk_size_respected() -> None:
    service = ChunkingService(chunk_size=20, chunk_overlap=0)
    text = "word " * 30  # longer than 20 chars
    pages = [ExtractedPage(page_number=1, text=text)]
    chunks = service.chunk_pages(pages, document_id="d1", filename="a.pdf")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)


def test_chunk_overlap_produces_shared_content() -> None:
    service = ChunkingService(chunk_size=20, chunk_overlap=8)
    # Continuous text without spaces so windowing is deterministic.
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    pages = [ExtractedPage(page_number=1, text=text)]
    chunks = service.chunk_pages(pages, document_id="d1", filename="a.pdf")

    assert len(chunks) >= 2
    assert chunks[0].text[-8:] == chunks[1].text[:8]


def test_chunk_metadata_preservation() -> None:
    service = ChunkingService(chunk_size=10, chunk_overlap=0)
    pages = [
        ExtractedPage(page_number=1, text="abcdefghijklmnop"),
        ExtractedPage(page_number=2, text=" stecondpagexx"),
    ]
    chunks = service.chunk_pages(pages, document_id="doc-9", filename="meta.pdf")

    assert chunks
    assert all(chunk.document_id == "doc-9" for chunk in chunks)
    assert all(chunk.filename == "meta.pdf" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_short_text_single_chunk() -> None:
    service = ChunkingService(chunk_size=100, chunk_overlap=10)
    pages = [ExtractedPage(page_number=1, text="short")]
    chunks = service.chunk_pages(pages, document_id="d", filename="f.pdf")
    assert len(chunks) == 1
    assert chunks[0].text == "short"
    assert chunks[0].chunk_index == 0


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="smaller"):
        ChunkingService(chunk_size=10, chunk_overlap=10)
