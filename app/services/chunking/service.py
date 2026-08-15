"""Text chunking service with configurable size and overlap."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.ingestion.pdf_extractor import ExtractedPage

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class TextChunk:
    """A single text chunk with preserved document metadata."""

    text: str
    chunk_index: int
    page_number: int
    document_id: str
    filename: str
    start_char: int
    end_char: int


class ChunkingService:
    """Split extracted page text into overlapping character windows."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            msg = "chunk_size must be > 0"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = "chunk_overlap must be >= 0"
            raise ValueError(msg)
        if chunk_overlap >= chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        *,
        document_id: str,
        filename: str,
    ) -> list[TextChunk]:
        """
        Chunk each page independently so page numbers stay accurate.

        Chunk indexes are assigned globally across the document (0-based).
        """
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue
            for text, start, end in self._split_text(page.text):
                chunks.append(
                    TextChunk(
                        text=text,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        document_id=document_id,
                        filename=filename,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_index += 1

        logger.info(
            "chunking_completed",
            extra={
                "operation": "chunk_pages",
                "document_id": document_id,
                "document_filename": filename,
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
            },
        )
        return chunks

    def _split_text(self, text: str) -> list[tuple[str, int, int]]:
        cleaned = text.strip()
        if not cleaned:
            return []

        # Map cleaned offsets back onto the original string when possible.
        origin = text.find(cleaned)
        if origin < 0:
            origin = 0
            working = cleaned
        else:
            working = text[origin : origin + len(cleaned)]

        if len(working) <= self._chunk_size:
            return [(working.strip(), origin, origin + len(working))]

        parts: list[tuple[str, int, int]] = []
        start = 0
        length = len(working)

        while start < length:
            end = min(start + self._chunk_size, length)
            if end < length:
                break_at = working.rfind(" ", start, end)
                if break_at > start:
                    end = break_at

            piece = working[start:end]
            stripped = piece.strip()
            if stripped:
                # Adjust absolute offsets to exclude leading/trailing whitespace in piece.
                leading = len(piece) - len(piece.lstrip())
                abs_start = origin + start + leading
                abs_end = abs_start + len(stripped)
                parts.append((stripped, abs_start, abs_end))

            if end >= length:
                break

            next_start = end - self._chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return parts
