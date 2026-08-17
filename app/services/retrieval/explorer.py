"""Retrieval explorer orchestration for exposing pipeline stages and raw results."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.schemas.retrieval import RetrievalMethod
from app.services.context_optimization.base import ContextOptimizer
from app.services.query_transformation.service import QueryTransformationService
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.keyword.base import KeywordSearch
from app.services.retrieval.multi_query import MultiQueryGenerator, MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class PipelineStage:
    """Internal pipeline stage record."""

    id: str
    label: str
    enabled: bool
    executed: bool
    result_count: int | None = None
    details: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class RetrievalExploreResult:
    """Internal retrieval explorer payload."""

    query: str
    retrieval_query: str
    generated_queries: list[str] | None
    configuration: dict[str, bool]
    pipeline: list[PipelineStage]
    vector_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk] | None
    results: list[RetrievedChunk]
    result_methods: dict[str, RetrievalMethod]
    metadata: dict[str, object]


class RetrievalExplorerService:
    """Run the configured retrieval pipeline and capture intermediate outputs."""

    def __init__(
        self,
        settings: Settings,
        vector_retrieval: RetrievalService,
        hybrid_retrieval: HybridRetrievalService,
        keyword_search: KeywordSearch,
        multi_query_retrieval: MultiQueryRetrievalService,
        *,
        query_transformer: QueryTransformationService | None = None,
        multi_query_generator: MultiQueryGenerator | None = None,
        context_optimizer: ContextOptimizer | None = None,
    ) -> None:
        self._settings = settings
        self._vector = vector_retrieval
        self._hybrid = hybrid_retrieval
        self._keyword = keyword_search
        self._multi_query = multi_query_retrieval
        self._query_transformer = query_transformer
        self._multi_query_generator = multi_query_generator
        self._context_optimizer = context_optimizer

    def explore(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalExploreResult:
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        pipeline: list[PipelineStage] = []
        metadata: dict[str, object] = {}

        pipeline.append(
            PipelineStage(
                id="query",
                label="Query",
                enabled=True,
                executed=True,
                details={"query": normalized},
            )
        )

        retrieval_query = normalized
        if self._query_transformer is not None:
            transformed = self._query_transformer.transform(normalized)
            retrieval_query = transformed.transformed_query
            pipeline.append(
                PipelineStage(
                    id="query_transformation",
                    label="Query Transformation",
                    enabled=True,
                    executed=True,
                    details={
                        "was_transformed": transformed.was_transformed,
                        "retrieval_query": retrieval_query,
                    },
                )
            )
        else:
            pipeline.append(
                PipelineStage(
                    id="query_transformation",
                    label="Query Transformation",
                    enabled=False,
                    executed=False,
                )
            )

        generated_queries: list[str] | None = None
        vector_results: list[RetrievedChunk] = []
        bm25_results: list[RetrievedChunk] = []
        fused_results: list[RetrievedChunk] | None = None
        retrieved: list[RetrievedChunk] = []
        result_methods: dict[str, RetrievalMethod] = {}

        limit = top_k if top_k is not None else self._default_top_k()
        intermediate_query = retrieval_query

        if self._settings.multi_query_enabled and self._multi_query_generator is not None:
            generated = self._multi_query_generator.generate(retrieval_query)
            generated_queries = list(generated.queries)
            pipeline.append(
                PipelineStage(
                    id="multi_query",
                    label="Multi-Query Generation",
                    enabled=True,
                    executed=True,
                    result_count=len(generated_queries),
                    details={"queries": generated_queries},
                )
            )
            if generated_queries:
                intermediate_query = generated_queries[0]
                metadata["intermediate_query_note"] = (
                    "Vector, BM25, and fusion previews use the first generated query."
                )

            try:
                retrieved = self._multi_query.retrieve(
                    retrieval_query,
                    top_k=top_k,
                    filters=filters,
                )
                combine_method = self._default_combine_method(generated_queries)
                for chunk in retrieved:
                    result_methods[chunk.chunk_id] = combine_method
                pipeline.append(
                    PipelineStage(
                        id="multi_query_combine",
                        label="Multi-Query Combine",
                        enabled=True,
                        executed=True,
                        result_count=len(retrieved),
                    )
                )
            except AppError:
                raise
        else:
            pipeline.append(
                PipelineStage(
                    id="multi_query",
                    label="Multi-Query Generation",
                    enabled=False,
                    executed=False,
                )
            )
            pipeline.append(
                PipelineStage(
                    id="multi_query_combine",
                    label="Multi-Query Combine",
                    enabled=False,
                    executed=False,
                )
            )
            retrieved, combine_method = self._retrieve_single_path(
                retrieval_query,
                top_k=top_k,
                filters=filters,
            )
            for chunk in retrieved:
                result_methods[chunk.chunk_id] = combine_method

        vector_results, bm25_results, fused_results = self._capture_hybrid_intermediates(
            intermediate_query,
            top_k=limit,
            filters=filters,
        )
        pipeline.extend(
            self._build_retrieval_stages(
                vector_count=len(vector_results),
                bm25_count=len(bm25_results),
                fused_count=len(fused_results) if fused_results is not None else None,
            )
        )

        pre_optimization_count = len(retrieved)
        if self._context_optimizer is not None:
            optimization = self._context_optimizer.optimize(retrieved)
            retrieved = optimization.chunks
            pipeline.append(
                PipelineStage(
                    id="context_optimization",
                    label="Context Optimization",
                    enabled=True,
                    executed=True,
                    result_count=len(retrieved),
                    details={
                        "input_count": pre_optimization_count,
                        "removed_count": optimization.removed_count,
                        "estimated_tokens": optimization.estimated_tokens,
                    },
                )
            )
            metadata["context_optimization_removed_count"] = optimization.removed_count
        else:
            pipeline.append(
                PipelineStage(
                    id="context_optimization",
                    label="Context Optimization",
                    enabled=False,
                    executed=False,
                )
            )

        pipeline.append(
            PipelineStage(
                id="final_results",
                label="Final Results",
                enabled=True,
                executed=True,
                result_count=len(retrieved),
            )
        )

        logger.info(
            "retrieval_explore_completed",
            extra={
                "operation": "explore_retrieval",
                "query_length": len(normalized),
                "result_count": len(retrieved),
                "hybrid_enabled": self._settings.hybrid_search_enabled,
                "multi_query_enabled": self._settings.multi_query_enabled,
            },
        )

        return RetrievalExploreResult(
            query=normalized,
            retrieval_query=retrieval_query,
            generated_queries=generated_queries,
            configuration={
                "query_transformation_enabled": self._query_transformer is not None,
                "multi_query_enabled": self._settings.multi_query_enabled,
                "hybrid_search_enabled": self._settings.hybrid_search_enabled,
                "context_optimization_enabled": self._context_optimizer is not None,
                "reranking_enabled": False,
            },
            pipeline=pipeline,
            vector_results=vector_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
            results=retrieved,
            result_methods=result_methods,
            metadata=metadata,
        )

    def _default_top_k(self) -> int:
        if self._settings.hybrid_search_enabled:
            return self._settings.hybrid_top_k
        return self._settings.retrieval_top_k

    def _default_combine_method(
        self,
        generated_queries: list[str] | None,
    ) -> RetrievalMethod:
        if (
            self._settings.multi_query_enabled
            and generated_queries is not None
            and len(generated_queries) > 1
        ):
            return "multi_query"
        if self._settings.hybrid_search_enabled:
            return "hybrid_fusion"
        return "vector"

    def _retrieve_single_path(
        self,
        query: str,
        *,
        top_k: int | None,
        filters: RetrievalFilters | None,
    ) -> tuple[list[RetrievedChunk], RetrievalMethod]:
        if self._settings.hybrid_search_enabled:
            chunks = self._hybrid.retrieve(query, top_k=top_k, filters=filters)
            return chunks, "hybrid_fusion"
        chunks = self._vector.retrieve(query, top_k=top_k, filters=filters)
        return chunks, "vector"

    def _capture_hybrid_intermediates(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[RetrievedChunk] | None]:
        vector_results = self._vector.retrieve(
            query,
            top_k=top_k,
            filters=filters,
        )
        if not self._settings.hybrid_search_enabled:
            return vector_results, [], None

        bm25_results = self._search_keywords(query, top_k=top_k, filters=filters)
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            weights=[
                self._settings.vector_search_weight,
                self._settings.keyword_search_weight,
            ],
            limit=top_k,
        )
        return vector_results, bm25_results, fused_results

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
                "retrieval_explore_keyword_failed",
                extra={
                    "operation": "explore_retrieval",
                    "error_type": type(exc).__name__,
                },
            )
            return []

    def _build_retrieval_stages(
        self,
        *,
        vector_count: int,
        bm25_count: int,
        fused_count: int | None,
    ) -> list[PipelineStage]:
        hybrid_enabled = self._settings.hybrid_search_enabled
        stages = [
            PipelineStage(
                id="vector_search",
                label="Vector Search",
                enabled=True,
                executed=True,
                result_count=vector_count,
            ),
            PipelineStage(
                id="bm25",
                label="BM25",
                enabled=hybrid_enabled,
                executed=hybrid_enabled,
                result_count=bm25_count if hybrid_enabled else None,
            ),
            PipelineStage(
                id="hybrid_fusion",
                label="Hybrid Fusion",
                enabled=hybrid_enabled,
                executed=hybrid_enabled,
                result_count=fused_count if hybrid_enabled else None,
                details={
                    "method": "reciprocal_rank_fusion",
                    "vector_weight": self._settings.vector_search_weight,
                    "keyword_weight": self._settings.keyword_search_weight,
                }
                if hybrid_enabled
                else None,
            ),
            PipelineStage(
                id="reranking",
                label="Reranking",
                enabled=False,
                executed=False,
                details={"reason": "not_configured"},
            ),
        ]
        return stages
