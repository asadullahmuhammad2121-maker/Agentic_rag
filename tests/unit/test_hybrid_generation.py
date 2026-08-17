"""Regression tests for hybrid agent answer generation (Phase 3F)."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.agent.generation.combined import merge_tool_outputs_to_chunks
from app.services.agent.models import (
    AgentObservation,
    AgentRequest,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
    TavilySearchOutput,
    WebSearchResultItem,
)
from app.services.agent.service import AgentService
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.agent.tools.tavily import TAVILY_WEB_SEARCH_TOOL_NAME
from app.services.llm.base import LLMService
from app.services.rag.prompt_builder import PromptBuilder
from app.services.rag.service import RAGService
from app.services.retrieval.service import RetrievedChunk

DIFFICULT_HYBRID_QUERY = (
    "According to my uploaded documents, explain RAG's architecture and limitations. "
    "Then compare those limitations with the latest RAG and agentic AI developments in 2026. "
    "For each major difference, explain whether the newer approach actually solves the "
    "limitation or only reduces it."
)


def _document_chunk(*, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc-1:{index:05d}",
        text=(
            "RAG architecture retrieves relevant chunks from a vector store, augments the "
            "prompt, and generates an answer. Limitations include retrieval quality, context "
            "window constraints, and stale knowledge."
        ),
        document_id="doc-1",
        filename="rag_guide.pdf",
        file_type="pdf",
        source="rag_guide.pdf",
        page_number=1,
        section=None,
        chunk_index=index,
        chunking_strategy="fixed",
        score=0.91,
    )


def _web_chunk(*, index: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="https://example.com/agentic-rag-2026",
        text=(
            "In 2026, agentic RAG systems combine routing, planning, hybrid retrieval, and "
            "tool use to reduce stale answers and improve multi-step reasoning."
        ),
        document_id="https://example.com/agentic-rag-2026",
        filename="Agentic RAG Trends 2026",
        file_type="web",
        source="https://example.com/agentic-rag-2026",
        page_number=0,
        section=None,
        chunk_index=index,
        chunking_strategy="web",
        score=0.87,
    )


def _rag_output() -> RAGRetrievalOutput:
    return RAGRetrievalOutput(
        query=DIFFICULT_HYBRID_QUERY,
        chunks=[
            RetrievedChunkOutput(
                chunk_id="doc-1:00000",
                text=_document_chunk().text,
                document_id="doc-1",
                filename="rag_guide.pdf",
                file_type="pdf",
                source="rag_guide.pdf",
                page_number=1,
                section=None,
                chunk_index=0,
                chunking_strategy="fixed",
                score=0.91,
            )
        ],
        result_count=1,
        empty=False,
    )


def _web_output() -> TavilySearchOutput:
    return TavilySearchOutput(
        query=DIFFICULT_HYBRID_QUERY,
        results=[
            WebSearchResultItem(
                title="Agentic RAG Trends 2026",
                url="https://example.com/agentic-rag-2026",
                content=_web_chunk().text,
                score=0.87,
            )
        ],
        result_count=1,
        empty=False,
    )


def test_combined_prompt_includes_document_and_web_sections() -> None:
    built = PromptBuilder().build_combined(
        DIFFICULT_HYBRID_QUERY,
        [_document_chunk(index=0), _web_chunk(index=1)],
    )

    assert "Uploaded document context:" in built.user_prompt
    assert "Web search results:" in built.user_prompt
    assert "rag_guide.pdf" in built.user_prompt
    assert "https://example.com/agentic-rag-2026" in built.user_prompt
    assert "[S1]" in built.user_prompt
    assert "[S2]" in built.user_prompt
    assert "knowledge base" not in built.system_prompt.lower()
    assert "web search results" in built.system_prompt.lower()


def test_generate_from_chunks_uses_combined_prompt_for_web_chunks() -> None:
    llm = MagicMock(spec=LLMService)
    llm.generate.return_value = (
        "RAG uses retrieval plus generation [S1]. Agentic RAG in 2026 adds planning and tools [S2]."
    )
    rag = RAGService(retrieval_service=MagicMock(), llm_service=llm)

    result = rag.generate_from_chunks(
        DIFFICULT_HYBRID_QUERY,
        [_document_chunk(index=0), _web_chunk(index=1)],
    )

    assert "Agentic RAG in 2026" in result.answer
    assert len(result.citations) == 2
    assert {citation.file_type for citation in result.citations} == {"pdf", "web"}
    llm.generate.assert_called_once()
    system_prompt = llm.generate.call_args.kwargs["system_prompt"]
    user_prompt = llm.generate.call_args.args[0]
    assert "web search results" in system_prompt.lower()
    assert "Uploaded document context:" in user_prompt
    assert "Web search results:" in user_prompt
    assert "do not have enough information in the knowledge base" not in system_prompt.lower()


def test_document_only_generation_keeps_rag_prompt() -> None:
    llm = MagicMock(spec=LLMService)
    llm.generate.return_value = "Document-only answer [S1]"
    rag = RAGService(retrieval_service=MagicMock(), llm_service=llm)

    rag.generate_from_chunks("What is RAG?", [_document_chunk(index=0)])

    system_prompt = llm.generate.call_args.kwargs["system_prompt"]
    assert "knowledge base" in system_prompt.lower()
    user_prompt = llm.generate.call_args.args[0]
    assert "Web search results:" not in user_prompt


def test_difficult_hybrid_query_generation_uses_both_sources() -> None:
    llm = MagicMock(spec=LLMService)
    llm.generate.return_value = (
        "Uploaded documents describe classic RAG architecture and limitations [S1]. "
        "Web results describe 2026 agentic RAG developments [S2]."
    )
    rag_service = RAGService(retrieval_service=MagicMock(), llm_service=llm)
    observation = AgentObservation(
        tool_name=f"{RAG_RETRIEVAL_TOOL_NAME}+{TAVILY_WEB_SEARCH_TOOL_NAME}",
        tool_names=[RAG_RETRIEVAL_TOOL_NAME, TAVILY_WEB_SEARCH_TOOL_NAME],
        success=True,
        metadata={
            "multi_tool": True,
            "tool_outputs": {
                RAG_RETRIEVAL_TOOL_NAME: _rag_output().model_dump(),
                TAVILY_WEB_SEARCH_TOOL_NAME: _web_output().model_dump(),
            },
        },
    )
    service = AgentService(
        agent=MagicMock(),
        tools=ToolRegistry([]),
        rag_service=rag_service,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )

    generated = service._generate_from_combined(
        AgentRequest(query=DIFFICULT_HYBRID_QUERY),
        observation,
    )

    merged = merge_tool_outputs_to_chunks(rag_output=_rag_output(), web_output=_web_output())
    assert len(merged) == 2
    assert "2026 agentic RAG" in generated.answer
    assert len(generated.citations) == 2
    assert {citation.file_type for citation in generated.citations} == {"pdf", "web"}
    assert "do not have enough information" not in generated.answer.lower()
    user_prompt = llm.generate.call_args.args[0]
    assert "Web search results:" in user_prompt
    assert "Uploaded document context:" in user_prompt
