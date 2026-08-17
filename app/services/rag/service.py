"""RAG orchestration: retrieve → prompt → generate → citations."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.services.context_optimization.base import ContextOptimizer
from app.services.llm.base import LLMService
from app.services.query_transformation.service import QueryTransformationService
from app.services.rag.prompt_builder import PromptBuilder
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.multi_query import MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk

logger = get_logger(__name__)

EMPTY_RETRIEVAL_ANSWER = (
    "I could not find relevant information in the knowledge base to answer that question."
)


@dataclass(slots=True, frozen=True)
class Citation:
    """Citation metadata for a retrieved chunk used in the answer."""

    document_id: str
    filename: str
    file_type: str
    source: str
    page_number: int
    section: str | None
    chunk_index: int
    chunk_id: str
    score: float
    label: str


@dataclass(slots=True, frozen=True)
class RAGResult:
    """Final RAG response payload."""

    answer: str
    citations: list[Citation]


@dataclass(slots=True, frozen=True)
class RetrievalContext:
    """Retrieved and optionally optimized chunks ready for generation."""

    query: str
    chunks: list[RetrievedChunk]


class RAGService:
    """Orchestrate Basic RAG: embed → retrieve → prompt → generate → format."""

    def __init__(
        self,
        retrieval_service: RetrievalService | MultiQueryRetrievalService,
        llm_service: LLMService,
        *,
        prompt_builder: PromptBuilder | None = None,
        query_transformer: QueryTransformationService | None = None,
        context_optimizer: ContextOptimizer | None = None,
    ) -> None:
        self._retrieval = retrieval_service
        self._llm = llm_service
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._query_transformer = query_transformer
        self._context_optimizer = context_optimizer

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RAGResult:
        """Run the full Advanced RAG pipeline: retrieve → prompt → generate."""
        context = self.retrieve_context(query, top_k=top_k, filters=filters)
        return self.generate_from_chunks(context.query, context.chunks)

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RetrievalContext:
        """
        Run the Advanced RAG retrieval pipeline without generation.

        Applies optional query transformation, retrieval, and context optimization.
        """
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        logger.info(
            "rag_retrieval_started",
            extra={
                "operation": "retrieve_context",
                "query_length": len(normalized),
                "top_k": top_k,
                "has_filters": filters is not None and not filters.is_empty(),
                "query_transformation_enabled": self._query_transformer is not None,
            },
        )

        retrieval_query = normalized
        if self._query_transformer is not None:
            transformed = self._query_transformer.transform(normalized)
            retrieval_query = transformed.transformed_query
            logger.info(
                "rag_query_transformation_applied",
                extra={
                    "operation": "retrieve_context",
                    "was_transformed": transformed.was_transformed,
                    "original_query_length": len(transformed.original_query),
                    "retrieval_query_length": len(retrieval_query),
                },
            )

        try:
            chunks = self._retrieval.retrieve(
                retrieval_query,
                top_k=top_k,
                filters=filters,
            )
        except AppError:
            raise

        if not chunks:
            logger.info(
                "rag_empty_retrieval",
                extra={"operation": "retrieve_context", "result_count": 0},
            )
            return RetrievalContext(query=normalized, chunks=[])

        if self._context_optimizer is not None:
            optimization = self._context_optimizer.optimize(chunks)
            chunks = optimization.chunks
            logger.info(
                "rag_context_optimized",
                extra={
                    "operation": "retrieve_context",
                    "removed_count": optimization.removed_count,
                    "estimated_tokens": optimization.estimated_tokens,
                    "result_count": len(chunks),
                },
            )
            if not chunks:
                logger.info(
                    "rag_empty_context_after_optimization",
                    extra={"operation": "retrieve_context", "result_count": 0},
                )
                return RetrievalContext(query=normalized, chunks=[])

        logger.info(
            "rag_retrieval_completed",
            extra={
                "operation": "retrieve_context",
                "result_count": len(chunks),
            },
        )
        return RetrievalContext(query=normalized, chunks=chunks)

    def generate_from_chunks(self, query: str, chunks: list[RetrievedChunk]) -> RAGResult:
        """Build a grounded prompt from retrieved chunks and generate an answer."""
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        if not chunks:
            logger.info(
                "rag_generation_skipped_empty_context",
                extra={"operation": "generate_from_chunks", "result_count": 0},
            )
            return RAGResult(answer=EMPTY_RETRIEVAL_ANSWER, citations=[])

        logger.info(
            "rag_generation_started",
            extra={
                "operation": "generate_from_chunks",
                "query_length": len(normalized),
                "result_count": len(chunks),
            },
        )

        prompt = self._prompt_builder.build(normalized, chunks)
        try:
            answer_text = self._llm.generate(
                prompt.user_prompt,
                system_prompt=prompt.system_prompt,
            )
        except AppError:
            raise

        citations = self._build_citations(chunks)
        logger.info(
            "rag_generation_completed",
            extra={
                "operation": "generate_from_chunks",
                "result_count": len(chunks),
                "citation_count": len(citations),
                "answer_length": len(answer_text),
            },
        )
        return RAGResult(answer=answer_text.strip(), citations=citations)

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for index, chunk in enumerate(chunks, start=1):
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    file_type=chunk.file_type,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    chunk_index=chunk.chunk_index,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                    label=f"S{index}",
                )
            )
        return citations
