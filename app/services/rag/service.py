"""RAG orchestration: retrieve → prompt → generate → citations."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import AppError, QueryError
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.services.rag.prompt_builder import PromptBuilder
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
    page_number: int
    chunk_index: int
    chunk_id: str
    score: float
    label: str


@dataclass(slots=True, frozen=True)
class RAGResult:
    """Final RAG response payload."""

    answer: str
    citations: list[Citation]


class RAGService:
    """Orchestrate Basic RAG: embed → retrieve → prompt → generate → format."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        *,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._retrieval = retrieval_service
        self._llm = llm_service
        self._prompt_builder = prompt_builder or PromptBuilder()

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, str | int] | None = None,
    ) -> RAGResult:
        normalized = query.strip()
        if not normalized:
            raise QueryError(
                "Query must not be empty",
                details={"reason": "empty_query"},
            )

        logger.info(
            "rag_query_started",
            extra={
                "operation": "rag_answer",
                "query_length": len(normalized),
                "top_k": top_k,
                "has_filters": bool(filters),
            },
        )

        try:
            chunks = self._retrieval.retrieve(
                normalized,
                top_k=top_k,
                filters=dict(filters) if filters else None,
            )
        except AppError:
            raise

        if not chunks:
            logger.info(
                "rag_empty_retrieval",
                extra={"operation": "rag_answer", "result_count": 0},
            )
            return RAGResult(answer=EMPTY_RETRIEVAL_ANSWER, citations=[])

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
            "rag_query_completed",
            extra={
                "operation": "rag_answer",
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
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    chunk_id=chunk.chunk_id,
                    score=chunk.score,
                    label=f"S{index}",
                )
            )
        return citations
