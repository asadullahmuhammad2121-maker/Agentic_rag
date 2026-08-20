"""Recursive chunking strategy."""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.chunking.base import Chunker, TextChunk, TextSegment
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.utils import (
    enforce_size_bounds,
    section_to_segment,
    segments_to_chunks,
    split_text_with_structured_blocks,
)
from app.services.ingestion.base import ExtractedSection

logger = get_logger(__name__)


class RecursiveChunker(Chunker):
    """Split text using progressively finer natural boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        *,
        document_id: str,
        filename: str,
        file_type: str,
        source: str,
    ) -> list[TextChunk]:
        segments: list[TextSegment] = []
        for section in sections:
            base = section_to_segment(section)
            if base is None:
                continue
            for text, start, end in split_text_with_structured_blocks(
                section.text,
                config=self._config,
                preserve_prose_intact=False,
            ):
                segments.append(
                    TextSegment(
                        text=text,
                        start_char=start,
                        end_char=end,
                        page_number=base.page_number,
                        section=base.section,
                    )
                )

        bounded = enforce_size_bounds(segments, config=self._config)
        chunks = segments_to_chunks(
            bounded,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            source=source,
        )
        logger.info(
            "chunking_completed",
            extra={
                "operation": "chunk_sections",
                "strategy": self.strategy_name,
                "document_id": document_id,
                "document_filename": filename,
                "file_type": file_type,
                "section_count": len(sections),
                "chunk_count": len(chunks),
            },
        )
        return chunks
