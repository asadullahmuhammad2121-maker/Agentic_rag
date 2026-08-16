"""Chunking service facade over configurable strategies."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.chunking.base import Chunker, TextChunk
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.factory import create_chunker
from app.services.embeddings.base import EmbeddingService
from app.services.ingestion.base import ExtractedPage, ExtractedSection

logger = get_logger(__name__)


class ChunkingService:
    """Delegate document splitting to the configured chunking strategy."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_service: EmbeddingService | None = None,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        effective_settings = settings
        if chunk_size is not None or chunk_overlap is not None:
            overrides: dict[str, int] = {}
            if chunk_size is not None:
                overrides["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                overrides["chunk_overlap"] = chunk_overlap
            effective_settings = settings.model_copy(update=overrides)

        self._config = ChunkingConfig.from_settings(effective_settings)
        self._chunker = chunker or create_chunker(
            effective_settings,
            embedding_service=embedding_service,
        )

    @property
    def strategy_name(self) -> str:
        return self._chunker.strategy_name

    @property
    def chunk_size(self) -> int:
        return self._config.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._config.chunk_overlap

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        *,
        document_id: str,
        filename: str,
        file_type: str | None = None,
        source: str | None = None,
    ) -> list[TextChunk]:
        return self._chunker.chunk_pages(
            pages,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            source=source,
        )

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        *,
        document_id: str,
        filename: str,
        file_type: str,
        source: str,
    ) -> list[TextChunk]:
        chunks = self._chunker.chunk_sections(
            sections,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            source=source,
        )
        logger.info(
            "chunking_service_completed",
            extra={
                "operation": "chunk_sections",
                "strategy": self.strategy_name,
                "document_id": document_id,
                "chunk_count": len(chunks),
            },
        )
        return chunks
