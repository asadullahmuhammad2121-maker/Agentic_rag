"""Converters between tool models and existing retrieval types."""

from __future__ import annotations

from dataclasses import asdict

from app.services.agent.models import (
    AgentCitation,
    AgentObservation,
    RetrievedChunkOutput,
    ToolResult,
)
from app.services.rag.service import Citation
from app.services.retrieval.service import RetrievedChunk


def chunk_to_output(chunk: RetrievedChunk) -> RetrievedChunkOutput:
    """Convert a retrieved chunk into structured tool output."""
    return RetrievedChunkOutput(
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
    )


def output_to_chunk(chunk: RetrievedChunkOutput) -> RetrievedChunk:
    """Convert structured tool output back into retrieval chunks."""
    return RetrievedChunk(
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
    )


def citations_from_rag(citations: list[Citation]) -> list[AgentCitation]:
    """Convert RAG citations into agent response citations."""
    return [AgentCitation.model_validate(asdict(citation)) for citation in citations]


def tool_result_to_observation(tool_name: str, result: ToolResult) -> AgentObservation:
    """Convert a structured tool result into an agent observation."""
    if result.success:
        assert result.output is not None
        return AgentObservation(
            tool_name=tool_name,
            success=True,
            tool_output=result.output.model_dump(),
            metadata={"result_count": _result_count(result.output)},
        )
    assert result.error is not None
    return AgentObservation(
        tool_name=tool_name,
        success=False,
        error=result.error.message,
        metadata={
            "error_code": result.error.code,
            **result.error.details,
        },
    )


def _result_count(output: object) -> int | None:
    result_count = getattr(output, "result_count", None)
    if isinstance(result_count, int):
        return result_count
    chunks = getattr(output, "chunks", None)
    if isinstance(chunks, list):
        return len(chunks)
    results = getattr(output, "results", None)
    if isinstance(results, list):
        return len(results)
    return None
