"""Hybrid vector + keyword retrieval."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.keyword.base import KeywordSearch
from app.services.retrieval.service import RetrievalService, RetrievedChunk

logger = get_logger(__name__)


class HybridRetrievalService:
    """Combine Qdrant vector search with BM25 keyword search."""

    def __init__(
        self,
        settings: Settings,
        vector_retrieval: RetrievalService,
        keyword_search: KeywordSearch,
    ) -> None:
        self._settings = settings
        self._vector = vector_retrieval
        self._keyword = keyword_search

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using vector-only or hybrid search."""
        if not self._settings.hybrid_search_enabled:
            return self._vector.retrieve(
                query,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold,
            )

        limit = top_k if top_k is not None else self._settings.hybrid_top_k
        candidate_limit = limit

        logger.info(
            "hybrid_retrieval_started",
            extra={
                "operation": "hybrid_retrieve",
                "top_k": limit,
                "has_filters": filters is not None and not filters.is_empty(),
            },
        )

        vector_results = self._vector.retrieve(
            query,
            top_k=candidate_limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        keyword_results = self._search_keywords(
            query,
            top_k=candidate_limit,
            filters=filters,
        )

        fused = reciprocal_rank_fusion(
            [vector_results, keyword_results],
            weights=[
                self._settings.vector_search_weight,
                self._settings.keyword_search_weight,
            ],
            limit=limit,
        )

        logger.info(
            "hybrid_retrieval_completed",
            extra={
                "operation": "hybrid_retrieve",
                "vector_count": len(vector_results),
                "keyword_count": len(keyword_results),
                "result_count": len(fused),
            },
        )
        return fused

    def _search_keywords(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None,
    ) -> list[RetrievedChunk]:
        try:
            return self._keyword.search(query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning(
                "keyword_search_failed",
                extra={
                    "operation": "hybrid_retrieve",
                    "error_type": type(exc).__name__,
                },
            )
            return []
