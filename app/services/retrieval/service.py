"""Semantic retrieval over the vector store."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import AppError, ProviderError, QueryError
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingService
from app.services.retrieval.chunk_mapping import payload_to_retrieved_chunk
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.models import RetrievedChunk
from app.vector_store.base import VectorStore

logger = get_logger(__name__)

__all__ = ["RetrievalService", "RetrievedChunk"]


class RetrievalService:
    """Embed a query and retrieve top-k similar chunks from Qdrant."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for ``query``.

        Returns an empty list when nothing matches (caller handles gracefully).
        """
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        limit = top_k if top_k is not None else self._settings.retrieval_top_k
        if limit <= 0:
            raise QueryError(
                "top_k must be greater than zero",
                details={"reason": "invalid_top_k", "top_k": limit},
            )

        threshold = (
            score_threshold
            if score_threshold is not None
            else self._settings.retrieval_score_threshold
        )

        payload_filter = filters.to_payload_filter() if filters else None

        logger.info(
            "retrieval_started",
            extra={
                "operation": "retrieve",
                "query_length": len(normalized),
                "top_k": limit,
                "has_filters": payload_filter is not None,
            },
        )

        try:
            query_vector = self._embedding_service.embed_query(normalized)
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "query_embedding_failed",
                extra={
                    "operation": "retrieve",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Failed to embed query for retrieval",
                provider=self._embedding_service.provider_name,
                details={"reason": "embedding_failed"},
            ) from exc

        try:
            hits = self._vector_store.search(
                self._settings.qdrant_collection_name,
                query_vector,
                limit=limit,
                score_threshold=threshold,
                filters=payload_filter,
            )
        except AppError:
            raise

        chunks = [
            payload_to_retrieved_chunk(hit.id, hit.score, hit.payload) for hit in hits
        ]
        logger.info(
            "retrieval_completed",
            extra={
                "operation": "retrieve",
                "top_k": limit,
                "result_count": len(chunks),
            },
        )
        return chunks
