"""Regression tests for RAG → document navigation recovery in the agent loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.agent.foundation import FoundationAgent
from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentStep,
    DocumentNavigationOutput,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
)
from app.services.agent.planning.planner import QueryPlanner
from app.services.agent.recovery.navigation import (
    is_insufficient_rag_answer,
    maybe_document_navigation_recovery,
)
from app.services.agent.routing.router import QueryRouter
from app.services.agent.service import AgentService
from app.services.agent.tools.document_navigation import (
    DOCUMENT_NAVIGATION_TOOL_NAME,
    DocumentNavigationTool,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME, RAGRetrievalTool
from app.services.agent.tools.registry import ToolRegistry
from app.services.rag.service import Citation, RAGResult, RetrievalContext
from app.services.retrieval.document_navigation import DocumentNavigationService
from app.services.retrieval.service import RetrievedChunk
from app.vector_store.base import SearchResult, VectorStore
from tests.conftest import make_settings


def _payload(
    *,
    document_id: str,
    chunk_index: int,
    chunk_id: str,
    text: str,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "page_number": 1,
        "text": text,
        "filename": "guide.pdf",
        "file_type": "pdf",
        "source": "guide.pdf",
        "chunking_strategy": "recursive",
    }


class _IndexedVectorStore:
    def __init__(self, records: dict[tuple[str, int], dict[str, Any]]) -> None:
        self._records = records

    def find_by_payload(
        self,
        collection_name: str,
        conditions: dict[str, Any],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        document_id = conditions.get("document_id")
        chunk_index = conditions.get("chunk_index")
        chunk_id = conditions.get("chunk_id")
        if chunk_id is not None and document_id is not None:
            for (doc_id, _index), payload in self._records.items():
                if doc_id == document_id and payload["chunk_id"] == chunk_id:
                    return [SearchResult(id=payload["chunk_id"], score=1.0, payload=payload)]
            return []
        if chunk_index is not None and document_id is not None:
            found = self._records.get((str(document_id), int(chunk_index)))
            if found is None:
                return []
            return [SearchResult(id=found["chunk_id"], score=1.0, payload=found)]
        return []


def _foundation_agent() -> FoundationAgent:
    settings = make_settings(agent_planning_enabled=False, agent_routing_enabled=False)
    llm = MagicMock()
    return FoundationAgent(QueryRouter(settings, llm), QueryPlanner(settings, llm), settings)


def _chunk_output(
    *,
    chunk_id: str,
    text: str,
    chunk_index: int,
    document_id: str = "doc-a",
) -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=chunk_index,
        chunking_strategy="recursive",
        score=0.9,
    )


def _recovery_service(
    *,
    rag_chunks: list[RetrievedChunk],
    nav_records: dict[tuple[str, int], dict[str, Any]],
    generation: MagicMock,
    max_steps: int = 2,
) -> AgentService:
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(
        query="pipeline stages",
        chunks=rag_chunks,
    )
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), _IndexedVectorStore(nav_records)),  # type: ignore[arg-type]
    )
    return AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag), nav_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=max_steps,
    )


def test_retrieval_navigation_recovery_generates_combined_answer() -> None:
    fragmented = RetrievedChunk(
        chunk_id="doc-a:00009",
        text="The Core Pipeline 1",
        document_id="doc-a",
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=9,
        chunking_strategy="recursive",
        score=0.91,
    )
    nav_records = {
        ("doc-a", 9): _payload(
            document_id="doc-a",
            chunk_index=9,
            chunk_id="doc-a:00009",
            text="The Core Pipeline 1",
        ),
        ("doc-a", 10): _payload(
            document_id="doc-a",
            chunk_index=10,
            chunk_id="doc-a:00010",
            text="Ingest: Documents are split into chunks.",
        ),
        ("doc-a", 11): _payload(
            document_id="doc-a",
            chunk_index=11,
            chunk_id="doc-a:00011",
            text="Store: Embeddings are saved in a vector database.",
        ),
    }
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I don't have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="The pipeline includes Ingest and Store stages.",
            citations=[
                Citation(
                    document_id="doc-a",
                    filename="guide.pdf",
                    file_type="pdf",
                    source="guide.pdf",
                    page_number=1,
                    section=None,
                    chunk_index=10,
                    chunk_id="doc-a:00010",
                    score=1.0,
                    label="S2",
                )
            ],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[fragmented],
        nav_records=nav_records,
        generation=generation,
    )

    result = service.run("What are the pipeline stages?")

    assert "Ingest" in result.answer
    assert generation.generate_from_chunks.call_count == 2
    second_chunks = generation.generate_from_chunks.call_args_list[1].args[1]
    texts = {chunk.text for chunk in second_chunks}
    assert "The Core Pipeline 1" in texts
    assert "Ingest: Documents are split into chunks." in texts
    assert result.steps[0].action.tool_name == RAG_RETRIEVAL_TOOL_NAME
    assert result.steps[1].action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME
    assert result.steps[1].observation is not None
    assert result.steps[1].observation.metadata.get("recovery") is True


def test_sufficient_rag_answer_skips_navigation_recovery() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc-a:00001",
        text="Complete answer context",
        document_id="doc-a",
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="recursive",
        score=0.95,
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="The complete grounded answer.",
        citations=[],
    )
    nav_records = {
        ("doc-a", 0): _payload(
            document_id="doc-a",
            chunk_index=0,
            chunk_id="doc-a:00001",
            text="Complete answer context",
        ),
    }
    service = _recovery_service(
        rag_chunks=[chunk],
        nav_records=nav_records,
        generation=generation,
    )

    result = service.run("What is in the document?")

    assert result.answer == "The complete grounded answer."
    assert generation.generate_from_chunks.call_count == 1
    assert len(result.steps) == 2
    assert result.steps[1].action.type is AgentActionType.FINISH
    assert result.metadata["finished"] is True


def test_max_steps_one_prevents_navigation_recovery() -> None:
    fragmented = RetrievedChunk(
        chunk_id="doc-a:00009",
        text="fragment",
        document_id="doc-a",
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=9,
        chunking_strategy="recursive",
        score=0.5,
    )
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="I don't have enough information in the provided context.",
        citations=[],
    )
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(
        query="pipeline",
        chunks=[fragmented],
    )
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag), nav_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )

    result = service.run("pipeline stages")

    assert "don't have enough information" in result.answer.casefold()
    assert generation.generate_from_chunks.call_count == 1
    assert all(step.action.tool_name != DOCUMENT_NAVIGATION_TOOL_NAME for step in result.steps)


def test_repeated_navigation_is_not_attempted() -> None:
    rag_observation_output = RAGRetrievalOutput(
        query="pipeline",
        chunks=[
            _chunk_output(chunk_id="doc-a:00009", text="fragment", chunk_index=9),
        ],
    )
    history = [
        AgentStep(
            action=AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                arguments={"query": "pipeline"},
            ),
            observation=AgentObservation(
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                success=True,
                tool_output=rag_observation_output.model_dump(),
                answer="I don't have enough information in the provided context.",
                metadata={"generated": True},
            ),
        ),
        AgentStep(
            action=AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=DOCUMENT_NAVIGATION_TOOL_NAME,
                tool_names=[DOCUMENT_NAVIGATION_TOOL_NAME],
                arguments={"document_id": "doc-a", "chunk_id": "doc-a:00009"},
            ),
            observation=AgentObservation(
                tool_name=DOCUMENT_NAVIGATION_TOOL_NAME,
                success=True,
                tool_output=DocumentNavigationOutput(
                    document_id="doc-a",
                    anchor_chunk_id="doc-a:00009",
                    anchor_chunk_index=9,
                    chunks=[],
                ).model_dump(),
                answer="I don't have enough information in the provided context.",
                metadata={"generated": True, "recovery": True},
            ),
        ),
    ]
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    recovery = maybe_document_navigation_recovery(
        history,
        tools=ToolRegistry([RAGRetrievalTool(MagicMock()), nav_tool]),
    )
    assert recovery is None


def test_missing_navigation_metadata_skips_recovery() -> None:
    history = [
        AgentStep(
            action=AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                arguments={"query": "pipeline"},
            ),
            observation=AgentObservation(
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                success=True,
                tool_output=RAGRetrievalOutput(
                    query="pipeline",
                    chunks=[
                        RetrievedChunkOutput(
                            chunk_id="",
                            text="fragment without ids",
                            document_id="",
                            filename="guide.pdf",
                            file_type="pdf",
                            source="guide.pdf",
                            page_number=1,
                            section=None,
                            chunk_index=9,
                            chunking_strategy="recursive",
                            score=0.5,
                        )
                    ],
                ).model_dump(),
                answer="I don't have enough information in the provided context.",
                metadata={"generated": True},
            ),
        )
    ]
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    recovery = maybe_document_navigation_recovery(
        history,
        tools=ToolRegistry([RAGRetrievalTool(MagicMock()), nav_tool]),
    )
    assert recovery is None


def test_is_insufficient_rag_answer_detects_refusals() -> None:
    assert is_insufficient_rag_answer("I don't have enough information in the provided context.")
    assert not is_insufficient_rag_answer("The pipeline has five stages.")
