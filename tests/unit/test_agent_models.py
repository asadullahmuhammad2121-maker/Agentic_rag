"""Unit tests for agent Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentRequest,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
    ToolError,
    ToolResult,
)


def test_agent_request_strips_query() -> None:
    request = AgentRequest(query="  What is RAG?  ")
    assert request.query == "What is RAG?"


def test_agent_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(query="   ")


def test_agent_request_tool_arguments_include_filters() -> None:
    request = AgentRequest(
        query="What is RAG?",
        top_k=3,
        document_ids=["doc-1"],
        filenames=["a.pdf"],
    )
    arguments = request.tool_arguments()
    assert arguments["query"] == "What is RAG?"
    assert arguments["top_k"] == 3
    assert arguments["document_ids"] == ["doc-1"]
    assert arguments["filenames"] == ["a.pdf"]
    assert "file_types" not in arguments


def test_call_tool_action_requires_tool_name() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=AgentActionType.CALL_TOOL, arguments={"query": "q"})


def test_finish_action_requires_answer() -> None:
    with pytest.raises(ValidationError):
        AgentAction(type=AgentActionType.FINISH)


def test_finish_action_allows_empty_answer() -> None:
    action = AgentAction(type=AgentActionType.FINISH, answer="")
    assert action.answer == ""
    assert action.type is AgentActionType.FINISH


class _SampleOutput(BaseModel):
    value: str


def test_tool_result_requires_output_on_success() -> None:
    with pytest.raises(ValidationError):
        ToolResult(success=True)


def test_tool_result_requires_error_on_failure() -> None:
    with pytest.raises(ValidationError):
        ToolResult(success=False)


def test_rag_retrieval_output_empty_flag() -> None:
    output = RAGRetrievalOutput(query="What is RAG?", chunks=[])
    assert output.empty is True
    assert output.result_count == 0


def test_retrieved_chunk_output_is_frozen() -> None:
    chunk = RetrievedChunkOutput(
        chunk_id="c1",
        text="hello",
        document_id="doc-1",
        filename="a.pdf",
        file_type="pdf",
        source="a.pdf",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        score=0.9,
    )
    with pytest.raises(ValidationError):
        chunk.chunk_id = "changed"  # type: ignore[misc]


def test_tool_error_carries_details() -> None:
    error = ToolError(code="invalid_input", message="bad args", details={"field": "query"})
    assert error.details["field"] == "query"
