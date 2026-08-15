"""RAG query routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import RAGServiceDep
from app.core.logging import get_logger
from app.schemas.query import CitationResponse, QueryRequest, QueryResponse

logger = get_logger(__name__)

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question over ingested documents",
)
def query_documents(
    body: QueryRequest,
    rag_service: RAGServiceDep,
) -> QueryResponse:
    """Run Basic RAG: retrieve → prompt → generate → cite."""
    logger.info(
        "query_request_received",
        extra={
            "operation": "query_documents",
            "query_length": len(body.query),
            "top_k": body.top_k,
            "has_filters": bool(body.filters),
        },
    )
    result = rag_service.answer(
        body.query,
        top_k=body.top_k,
        filters=body.filters,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                document_id=citation.document_id,
                filename=citation.filename,
                page_number=citation.page_number,
                chunk_index=citation.chunk_index,
                chunk_id=citation.chunk_id,
                score=citation.score,
                label=citation.label,
            )
            for citation in result.citations
        ],
        metadata={"citation_count": len(result.citations)},
    )
