"""Internal RAG retrieval tool — wraps the existing Advanced RAG retrieval pipeline."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.agent.models import RAGRetrievalInput, RAGRetrievalOutput, ToolResult
from app.services.agent.tools.base import Tool
from app.services.agent.tools.converters import chunk_to_output
from app.services.rag.service import RAGService
from app.services.retrieval.filters import RetrievalFilters

logger = get_logger(__name__)

RAG_RETRIEVAL_TOOL_NAME = "rag_retrieval"


class RAGRetrievalTool(Tool):
    """Retrieve relevant document chunks using the existing Advanced RAG pipeline."""

    def __init__(self, rag_service: RAGService) -> None:
        self._rag = rag_service

    @property
    def name(self) -> str:
        return RAG_RETRIEVAL_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Retrieve relevant document chunks from the internal knowledge base. "
            "Use this for questions that can be answered from ingested documents."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return RAGRetrievalInput

    @property
    def output_model(self) -> type[BaseModel]:
        return RAGRetrievalOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        payload = RAGRetrievalInput.model_validate(validated_input.model_dump())
        filters = RetrievalFilters.from_query(
            document_ids=payload.document_ids,
            filenames=payload.filenames,
            file_types=payload.file_types,
            sections=payload.sections,
            legacy_filters=payload.filters,
        )

        logger.info(
            "rag_retrieval_tool_started",
            extra={
                "operation": "rag_retrieval_tool",
                "query_length": len(payload.query),
                "top_k": payload.top_k,
                "has_filters": filters is not None,
            },
        )

        context = self._rag.retrieve_context(
            payload.query,
            top_k=payload.top_k,
            filters=filters,
        )
        output = RAGRetrievalOutput(
            query=context.query,
            chunks=[chunk_to_output(chunk) for chunk in context.chunks],
        )

        logger.info(
            "rag_retrieval_tool_completed",
            extra={
                "operation": "rag_retrieval_tool",
                "result_count": output.result_count,
                "empty": output.empty,
            },
        )
        return ToolResult(success=True, output=output)
