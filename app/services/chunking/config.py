"""Chunking configuration derived from application settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings

ChunkingStrategy = Literal["fixed", "recursive", "semantic", "structure"]


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Validated chunking parameters shared by all strategies."""

    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int
    semantic_similarity_threshold: float

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingConfig:
        return cls(
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.chunk_min_size,
            max_chunk_size=settings.chunk_max_size,
            semantic_similarity_threshold=settings.semantic_similarity_threshold,
        )
