"""Chunker abstraction and shared chunk model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.services.ingestion.base import ExtractedPage, ExtractedSection


@dataclass(slots=True, frozen=True)
class TextChunk:
    """A single text chunk with preserved document metadata."""

    text: str
    chunk_id: str
    chunk_index: int
    page_number: int
    document_id: str
    filename: str
    start_char: int
    end_char: int
    file_type: str | None = None
    section: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class TextSegment:
    """Intermediate split result before chunk metadata is attached."""

    text: str
    start_char: int
    end_char: int
    page_number: int
    section: str | None


class Chunker(ABC):
    """Strategy interface for splitting normalized document sections."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the configured strategy identifier."""

    @abstractmethod
    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        *,
        document_id: str,
        filename: str,
        file_type: str,
        source: str,
    ) -> list[TextChunk]:
        """Split sections into retrieval-ready chunks."""

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        *,
        document_id: str,
        filename: str,
        file_type: str | None = None,
        source: str | None = None,
    ) -> list[TextChunk]:
        """Backward-compatible PDF page chunking entry point."""
        normalized_sections = [
            ExtractedSection(
                text=page.text,
                section_index=index,
                page_number=page.page_number,
                section=None,
            )
            for index, page in enumerate(pages)
        ]
        return self.chunk_sections(
            normalized_sections,
            document_id=document_id,
            filename=filename,
            file_type=file_type or "pdf",
            source=source or filename,
        )
