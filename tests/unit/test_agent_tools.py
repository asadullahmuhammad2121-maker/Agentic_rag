"""Unit tests for agent tools and registry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from app.core.exceptions import ConfigurationError, QdrantConnectionError, QueryError
from app.services.agent.models import RAGRetrievalInput, RAGRetrievalOutput, ToolResult
from app.services.agent.tools.base import Tool
from app.services.agent.tools.converters import tool_result_to_observation
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.rag.service import RetrievalContext
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.service import RetrievedChunk


class _SampleInput(BaseModel):
    query: str


class _SampleOutput(BaseModel):
    answer: str


class _NamedTool(Tool):
    def __init__(self, name: str, *, answer: str = "ok") -> None:
        self._name = name
        self._answer = answer

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test tool"

    @property
    def input_model(self) -> type[BaseModel]:
        return _SampleInput

    @property
    def output_model(self) -> type[BaseModel]:
        return _SampleOutput

    def execute(self, validated_input: BaseModel) -> ToolResult:
        payload = _SampleInput.model_validate(validated_input.model_dump())
        return ToolResult(success=True, output=_SampleOutput(answer=self._answer + payload.query))


def test_registry_registers_and_looks_up_tools() -> None:
    registry = ToolRegistry([_NamedTool("alpha")])
    assert "alpha" in registry
    assert registry.names() == ["alpha"]
    assert registry.get("alpha") is not None
    assert registry.get("missing") is None


def test_registry_lists_available_tools() -> None:
    registry = ToolRegistry([_NamedTool("alpha"), _NamedTool("beta")])
    tools = registry.list_tools()
    assert [tool.name for tool in tools] == ["alpha", "beta"]
    assert tools[0].description == "test tool"


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry([_NamedTool("alpha")])
    with pytest.raises(ConfigurationError) as exc_info:
        registry.register(_NamedTool("alpha"))
    assert exc_info.value.details.get("reason") == "duplicate_tool"


def test_registry_rejects_empty_tool_name() -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        ToolRegistry([_NamedTool("   ")])
    assert exc_info.value.details.get("reason") == "invalid_tool_name"


def test_named_tool_run_validates_input() -> None:
    tool = _NamedTool("alpha")
    with pytest.raises(QueryError) as exc_info:
        tool.run({})
    assert exc_info.value.details.get("reason") == "invalid_tool_input"


def test_named_tool_run_returns_structured_output() -> None:
    tool = _NamedTool("alpha", answer="done:")
    result = tool.run({"query": "hello"})
    assert result.success is True
    assert isinstance(result.output, _SampleOutput)
    assert result.output.answer == "done:hello"


def test_tool_result_to_observation_success() -> None:
    result = ToolResult(success=True, output=RAGRetrievalOutput(query="q", chunks=[]))
    observation = tool_result_to_observation(RAG_RETRIEVAL_TOOL_NAME, result)
    assert observation.success is True
    assert observation.tool_output == {"query": "q", "chunks": []}
    assert observation.metadata["result_count"] == 0


def _sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        text="Cats are mammals.",
        document_id="doc-1",
        filename="animals.pdf",
        file_type="pdf",
        source="animals.pdf",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        score=0.95,
    )


def test_rag_tool_delegates_to_retrieve_context() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(query="What are cats?", chunks=[_sample_chunk()])
    tool = RAGRetrievalTool(rag)

    assert tool.name == RAG_RETRIEVAL_TOOL_NAME
    result = tool.run({"query": "What are cats?", "top_k": 4})

    assert result.success is True
    assert isinstance(result.output, RAGRetrievalOutput)
    assert result.output.result_count == 1
    assert result.output.chunks[0].filename == "animals.pdf"
    rag.retrieve_context.assert_called_once()
    assert rag.retrieve_context.call_args.args[0] == "What are cats?"
    assert rag.retrieve_context.call_args.kwargs["top_k"] == 4
    assert rag.retrieve_context.call_args.kwargs["filters"] is None
    rag.generate_from_chunks.assert_not_called()


def test_rag_tool_passes_retrieval_filters() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(query="filtered?", chunks=[])
    tool = RAGRetrievalTool(rag)

    tool.run({"query": "filtered?", "document_ids": ["doc-1"]})

    filters = rag.retrieve_context.call_args.kwargs["filters"]
    assert isinstance(filters, RetrievalFilters)
    assert filters.document_ids == ("doc-1",)


def test_rag_tool_rejects_invalid_arguments() -> None:
    tool = RAGRetrievalTool(MagicMock())
    with pytest.raises(QueryError) as exc_info:
        tool.run({"query": "   "})
    assert exc_info.value.details.get("reason") == "invalid_tool_input"


def test_rag_tool_propagates_retrieval_failures() -> None:
    rag = MagicMock()
    rag.retrieve_context.side_effect = QdrantConnectionError()
    tool = RAGRetrievalTool(rag)
    with pytest.raises(QdrantConnectionError):
        tool.run({"query": "What is RAG?"})


def test_rag_tool_returns_empty_retrieval_output() -> None:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(query="unknown", chunks=[])
    tool = RAGRetrievalTool(rag)

    result = tool.run({"query": "unknown"})

    assert result.success is True
    assert isinstance(result.output, RAGRetrievalOutput)
    assert result.output.empty is True


def test_rag_tool_input_model_matches_arguments() -> None:
    tool = RAGRetrievalTool(MagicMock())
    assert tool.input_model is RAGRetrievalInput
    assert tool.output_model is RAGRetrievalOutput
