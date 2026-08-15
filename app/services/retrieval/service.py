"""Semantic retrieval over the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingService
from app.vector_store.base import VectorStore

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """A retrieved chunk with citation-ready metadata."""

    chunk_id: str
    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float


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
        filters: dict[str, Any] | None = None,
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

        logger.info(
            "retrieval_started",
            extra={
                "operation": "retrieve",
                "query_length": len(normalized),
                "top_k": limit,
                "has_filters": bool(filters),
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
            raise

        try:
            hits = self._vector_store.search(
                self._settings.qdrant_collection_name,
                query_vector,
                limit=limit,
                score_threshold=threshold,
                filters=filters,
            )
        except AppError:
            raise

        chunks = [self._to_chunk(hit.id, hit.score, hit.payload) for hit in hits]
        logger.info(
            "retrieval_completed",
            extra={
                "operation": "retrieve",
                "top_k": limit,
                "result_count": len(chunks),
            },
        )
        return chunks

    def _to_chunk(self, point_id: str, score: float, payload: dict[str, Any]) -> RetrievedChunk:
        page_number = payload.get("page_number", 0)
        chunk_index = payload.get("chunk_index", 0)
        try:
            page_number_int = int(page_number)
        except (TypeError, ValueError):
            page_number_int = 0
        try:
            chunk_index_int = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index_int = 0

        return RetrievedChunk(
            chunk_id=str(point_id),
            text=str(payload.get("text", "")),
            document_id=str(payload.get("document_id", "")),
            filename=str(payload.get("filename", "")),
            page_number=page_number_int,
            chunk_index=chunk_index_int,
            score=float(score),
        )
