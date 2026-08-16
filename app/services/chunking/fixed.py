"""Fixed-size chunking strategy."""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.chunking.base import Chunker, TextChunk
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.utils import (
    enforce_size_bounds,
    fixed_split_text,
    section_to_segment,
    segments_to_chunks,
)
from app.services.ingestion.base import ExtractedSection

logger = get_logger(__name__)


class FixedSizeChunker(Chunker):
    """Split text into overlapping character windows."""

    def __init__(self, config: ChunkingConfig) -> None:
        if config.chunk_overlap >= config.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(msg)
        self._config = config

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        *,
        document_id: str,
        filename: str,
        file_type: str,
        source: str,
    ) -> list[TextChunk]:
        segments = []
        for section in sections:
            base = section_to_segment(section)
            if base is None:
                continue
            for text, start, end in fixed_split_text(section.text, config=self._config):
                segments.append(
                    type(base)(
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
                "chunk_size": self._config.chunk_size,
                "chunk_overlap": self._config.chunk_overlap,
            },
        )
        return chunks
