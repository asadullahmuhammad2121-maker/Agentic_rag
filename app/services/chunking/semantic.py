"""Semantic chunking strategy using embedding similarity breakpoints."""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.chunking.base import Chunker, TextChunk, TextSegment
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.utils import (
    cosine_similarity,
    enforce_size_bounds,
    section_to_segment,
    segments_to_chunks,
    split_sentences,
)
from app.services.embeddings.base import EmbeddingService
from app.services.ingestion.base import ExtractedSection

logger = get_logger(__name__)


class SemanticChunker(Chunker):
    """Group semantically similar adjacent sentences into chunks."""

    def __init__(
        self,
        config: ChunkingConfig,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config = config
        self._embedding_service = embedding_service

    @property
    def strategy_name(self) -> str:
        return "semantic"

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
            segments.extend(self._chunk_section_semantically(section, base))

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

    def _chunk_section_semantically(
        self,
        section: ExtractedSection,
        base: TextSegment,
    ) -> list[TextSegment]:
        sentences = split_sentences(section.text)
        if len(sentences) == 1:
            return [base]

        vectors = self._embedding_service.embed_documents(sentences)
        if len(vectors) != len(sentences):
            return [base]

        groups: list[list[str]] = [[sentences[0]]]
        for index in range(1, len(sentences)):
            current = sentences[index]
            candidate = " ".join([*groups[-1], current]).strip()
            similarity = cosine_similarity(vectors[index - 1], vectors[index])
            if (
                similarity < self._config.semantic_similarity_threshold
                or len(candidate) > self._config.max_chunk_size
            ):
                groups.append([current])
            else:
                groups[-1].append(current)

        segments: list[TextSegment] = []
        cursor = base.start_char
        for group in groups:
            text = " ".join(group).strip()
            if not text:
                continue
            origin = section.text.find(text, cursor - base.start_char)
            if origin < 0:
                origin = section.text.find(text)
            start = cursor if origin < 0 else origin
            end = start + len(text)
            segments.append(
                TextSegment(
                    text=text,
                    start_char=start,
                    end_char=end,
                    page_number=base.page_number,
                    section=base.section,
                )
            )
            cursor = end
        return segments or [base]
