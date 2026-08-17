"""Retrieval explorer routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import RetrievalExplorerServiceDep
from app.core.logging import get_logger
from app.schemas.retrieval import (
    PipelineStageResponse,
    RetrievalConfigurationResponse,
    RetrievalExploreRequest,
    RetrievalExploreResponse,
    RetrievalMethod,
    RetrievedChunkResponse,
)
from app.services.retrieval.explorer import RetrievalExploreResult
from app.services.retrieval.service import RetrievedChunk

logger = get_logger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def _chunk_to_response(
    chunk: RetrievedChunk,
    *,
    retrieval_method: RetrievalMethod,
) -> RetrievedChunkResponse:
    return RetrievedChunkResponse(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_id=chunk.document_id,
        filename=chunk.filename,
        file_type=chunk.file_type,
        source=chunk.source,
        page_number=chunk.page_number,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
        chunking_strategy=chunk.chunking_strategy,
        score=chunk.score,
        retrieval_method=retrieval_method,
    )


def _to_response(result: RetrievalExploreResult) -> RetrievalExploreResponse:
    method_for = result.result_methods

    def map_chunks(
        chunks: list[RetrievedChunk],
        default_method: RetrievalMethod,
    ) -> list[RetrievedChunkResponse]:
        return [
            _chunk_to_response(
                chunk,
                retrieval_method=method_for.get(chunk.chunk_id, default_method),
            )
            for chunk in chunks
        ]

    return RetrievalExploreResponse(
        query=result.query,
        retrieval_query=result.retrieval_query,
        generated_queries=result.generated_queries,
        configuration=RetrievalConfigurationResponse(**result.configuration),
        pipeline=[
            PipelineStageResponse(
                id=stage.id,
                label=stage.label,
                enabled=stage.enabled,
                executed=stage.executed,
                result_count=stage.result_count,
                details=dict(stage.details or {}),
            )
            for stage in result.pipeline
        ],
        vector_results=map_chunks(result.vector_results, "vector"),
        bm25_results=map_chunks(result.bm25_results, "bm25"),
        fused_results=(
            map_chunks(result.fused_results, "hybrid_fusion")
            if result.fused_results is not None
            else None
        ),
        results=[
            _chunk_to_response(
                chunk,
                retrieval_method=method_for.get(chunk.chunk_id, "vector"),
            )
            for chunk in result.results
        ],
        metadata=dict(result.metadata),
    )


@router.post(
    "/explore",
    response_model=RetrievalExploreResponse,
    status_code=status.HTTP_200_OK,
    summary="Explore the retrieval pipeline and return raw chunk results",
)
def explore_retrieval(
    body: RetrievalExploreRequest,
    explorer_service: RetrievalExplorerServiceDep,
) -> RetrievalExploreResponse:
    """Run retrieval without generation and expose pipeline stages and chunk payloads."""
    retrieval_filters = body.build_retrieval_filters()
    logger.info(
        "retrieval_explore_request_received",
        extra={
            "operation": "explore_retrieval",
            "query_length": len(body.query),
            "top_k": body.top_k,
            "has_filters": retrieval_filters is not None,
        },
    )
    result = explorer_service.explore(
        body.query,
        top_k=body.top_k,
        filters=retrieval_filters,
    )
    return _to_response(result)
