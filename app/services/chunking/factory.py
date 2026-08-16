"""Factory for configured chunking strategies."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.services.chunking.base import Chunker
from app.services.chunking.config import ChunkingConfig
from app.services.chunking.fixed import FixedSizeChunker
from app.services.chunking.recursive import RecursiveChunker
from app.services.chunking.semantic import SemanticChunker
from app.services.chunking.structure import StructureAwareChunker
from app.services.embeddings.base import EmbeddingService


def create_chunker(
    settings: Settings,
    *,
    embedding_service: EmbeddingService | None = None,
) -> Chunker:
    """Build the configured chunking strategy."""
    config = ChunkingConfig.from_settings(settings)
    match config.strategy:
        case "fixed":
            return FixedSizeChunker(config)
        case "recursive":
            return RecursiveChunker(config)
        case "semantic":
            if embedding_service is None:
                raise ConfigurationError(
                    "Semantic chunking requires an embedding service",
                    details={"chunking_strategy": config.strategy},
                )
            return SemanticChunker(config, embedding_service)
        case "structure":
            return StructureAwareChunker(config)
        case _:
            raise ConfigurationError(
                "Unsupported chunking strategy",
                details={"chunking_strategy": config.strategy},
            )
