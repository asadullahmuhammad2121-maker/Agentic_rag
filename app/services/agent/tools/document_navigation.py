"""Document navigation tool — retrieve nearby chunks from the same document."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.agent.models import DocumentNavigationInput, DocumentNavigationOutput, ToolResult
from app.services.agent.tools.base import Tool
from app.services.agent.tools.converters import chunk_to_output
from app.services.retrieval.document_navigation import DocumentNavigationService

logger = get_logger(__name__)

DOCUMENT_NAVIGATION_TOOL_NAME = "document_navigation"


class DocumentNavigationTool(Tool):
    """Retrieve neighboring chunks from the same ingested document."""

    def __init__(self, navigation_service: DocumentNavigationService) -> None:
        self._navigation = navigation_service

    @property
    def name(self) -> str:
        return DOCUMENT_NAVIGATION_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Retrieve additional nearby chunks from the same ingested document when "
            "retrieved context appears incomplete. Uses document_id with chunk_id, "
            "chunk_index, or page_number metadata. Does not perform a new semantic search."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return DocumentNavigationInput

    @property
    def output_model(self) -> type[BaseModel]:
        return DocumentNavigationOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        payload = DocumentNavigationInput.model_validate(validated_input.model_dump())
        logger.info(
            "document_navigation_tool_started",
            extra={
                "operation": "document_navigation_tool",
                "document_id": payload.document_id,
                "chunk_id": payload.chunk_id,
                "chunk_index": payload.chunk_index,
                "page_number": payload.page_number,
            },
        )
        result = self._navigation.navigate(
            document_id=payload.document_id,
            chunk_id=payload.chunk_id,
            chunk_index=payload.chunk_index,
            page_number=payload.page_number,
            window=payload.window,
            limit=payload.limit,
        )
        output = DocumentNavigationOutput(
            document_id=result.document_id,
            anchor_chunk_id=result.anchor_chunk_id,
            anchor_chunk_index=result.anchor_chunk_index,
            chunks=[chunk_to_output(chunk) for chunk in result.chunks],
        )
        logger.info(
            "document_navigation_tool_completed",
            extra={
                "operation": "document_navigation_tool",
                "document_id": output.document_id,
                "result_count": output.result_count,
                "empty": output.empty,
            },
        )
        return ToolResult(success=True, output=output)
