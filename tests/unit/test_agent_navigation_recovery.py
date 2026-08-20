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
    retrieval_likely_incomplete_for_query,
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

CORE_PIPELINE_STAGES = ("Ingest", "Store", "Retrieve", "Augment", "Generate")

FIVE_STAGES_QUERY = (
    "According to my uploaded RAG document, what are the five stages of the Core Pipeline?"
)
STAGE_AFTER_INGEST_QUERY = (
    "According to my uploaded RAG document, what is the stage immediately after Ingest "
    "in the Core Pipeline?"
)
PIPELINE_FROM_INGEST_QUERY = (
    "According to my uploaded RAG document, explain the Core Pipeline starting from Ingest "
    "and include the next stages in order."
)


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


def _core_pipeline_nav_records() -> dict[tuple[str, int], dict[str, Any]]:
    stage_text = {
        "Ingest": "Ingest: Documents are split into chunks and converted into vector embeddings.",
        "Store": "Store: Embeddings are saved in a vector database.",
        "Retrieve": "Retrieve: A user query is embedded and matched against stored vectors.",
        "Augment": "Augment: Retrieved chunks are inserted into the prompt as context.",
        "Generate": "Generate: The language model produces a response grounded in that context.",
    }
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for index, stage in enumerate(CORE_PIPELINE_STAGES):
        chunk_index = 9 + index
        records[("doc-a", chunk_index)] = _payload(
            document_id="doc-a",
            chunk_index=chunk_index,
            chunk_id=f"doc-a:{chunk_index:05d}",
            text=(f"Core Pipeline\n{index + 1}\n{stage_text[stage]}"),
        )
    return records


def _fragmented_ingest_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc-a:00009",
        text="Core Pipeline\n1\nIngest: Documents are split into chunks.",
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


def _complete_pipeline_chunk() -> RetrievedChunk:
    combined = "\n".join(
        f"{index + 1}\n{stage}: stage description."
        for index, stage in enumerate(CORE_PIPELINE_STAGES)
    )
    return RetrievedChunk(
        chunk_id="doc-a:00009",
        text=f"Core Pipeline\n{combined}",
        document_id="doc-a",
        filename="guide.pdf",
        file_type="pdf",
        source="guide.pdf",
        page_number=1,
        section=None,
        chunk_index=9,
        chunking_strategy="recursive",
        score=0.95,
    )


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


def _full_pipeline_answer() -> str:
    return "The five Core Pipeline stages are Ingest, Store, Retrieve, Augment, and Generate."


def _full_pipeline_citations() -> list[Citation]:
    return [
        Citation(
            document_id="doc-a",
            filename="guide.pdf",
            file_type="pdf",
            source="guide.pdf",
            page_number=1,
            section=None,
            chunk_index=9 + index,
            chunk_id=f"doc-a:{9 + index:05d}",
            score=1.0,
            label=f"S{index + 1}",
        )
        for index in range(len(CORE_PIPELINE_STAGES))
    ]


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


def _assert_pipeline_stages(answer: str) -> None:
    for stage in CORE_PIPELINE_STAGES:
        assert stage in answer


def test_five_stages_query_triggers_navigation_and_combined_answer() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="The Core Pipeline begins with Ingest.",
            citations=[],
        ),
        RAGResult(
            answer=_full_pipeline_answer(),
            citations=_full_pipeline_citations(),
        ),
    ]
    service = _recovery_service(
        rag_chunks=[_fragmented_ingest_chunk()],
        nav_records=_core_pipeline_nav_records(),
        generation=generation,
    )

    result = service.run(FIVE_STAGES_QUERY)

    _assert_pipeline_stages(result.answer)
    assert generation.generate_from_chunks.call_count == 2
    assert result.steps[0].action.tool_name == RAG_RETRIEVAL_TOOL_NAME
    assert result.steps[1].action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME
    assert result.steps[1].observation is not None
    assert result.steps[1].observation.metadata.get("recovery") is True
    assert result.citations


def test_stage_after_ingest_query_triggers_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="Ingest is the first Core Pipeline stage.",
            citations=[],
        ),
        RAGResult(
            answer="The stage immediately after Ingest is Store.",
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
        rag_chunks=[_fragmented_ingest_chunk()],
        nav_records=_core_pipeline_nav_records(),
        generation=generation,
    )

    result = service.run(STAGE_AFTER_INGEST_QUERY)

    assert "Store" in result.answer
    assert generation.generate_from_chunks.call_count == 2
    assert result.steps[1].action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME


def test_pipeline_from_ingest_query_triggers_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="Ingest splits documents into chunks.",
            citations=[],
        ),
        RAGResult(
            answer=(
                "Starting from Ingest, the Core Pipeline continues with Store, "
                "Retrieve, Augment, and Generate in order."
            ),
            citations=_full_pipeline_citations(),
        ),
    ]
    service = _recovery_service(
        rag_chunks=[_fragmented_ingest_chunk()],
        nav_records=_core_pipeline_nav_records(),
        generation=generation,
    )

    result = service.run(PIPELINE_FROM_INGEST_QUERY)

    assert "Ingest" in result.answer
    assert "Store" in result.answer
    assert "Retrieve" in result.answer
    assert generation.generate_from_chunks.call_count == 2
    assert result.steps[1].action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME


def test_partial_context_without_refusal_triggers_navigation() -> None:
    retrieval = RAGRetrievalOutput(
        query="pipeline",
        chunks=[
            _chunk_output(
                chunk_id="doc-a:00009", text=_fragmented_ingest_chunk().text, chunk_index=9
            )
        ],
    )
    assert retrieval_likely_incomplete_for_query(FIVE_STAGES_QUERY, retrieval)

    recovery = maybe_document_navigation_recovery(
        [
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                    arguments={"query": FIVE_STAGES_QUERY},
                ),
                observation=AgentObservation(
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    success=True,
                    tool_output=retrieval.model_dump(),
                    answer="The Core Pipeline begins with Ingest.",
                    metadata={"generated": True},
                ),
            )
        ],
        query=FIVE_STAGES_QUERY,
        tools=ToolRegistry(
            [
                RAGRetrievalTool(MagicMock()),
                DocumentNavigationTool(
                    DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
                ),
            ]
        ),
    )
    assert recovery is not None
    assert recovery.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME


def test_retrieval_navigation_recovery_generates_combined_answer() -> None:
    nav_records = _core_pipeline_nav_records()
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
        rag_chunks=[_fragmented_ingest_chunk()],
        nav_records=nav_records,
        generation=generation,
    )

    result = service.run("What are the pipeline stages?")

    assert "Ingest" in result.answer
    assert generation.generate_from_chunks.call_count == 2
    second_chunks = generation.generate_from_chunks.call_args_list[1].args[1]
    texts = {chunk.text for chunk in second_chunks}
    assert "Ingest:" in next(iter(texts))
    assert result.steps[1].action.tool_name == DOCUMENT_NAVIGATION_TOOL_NAME


def test_sufficient_rag_context_skips_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer=_full_pipeline_answer(),
        citations=_full_pipeline_citations(),
    )
    service = _recovery_service(
        rag_chunks=[_complete_pipeline_chunk()],
        nav_records=_core_pipeline_nav_records(),
        generation=generation,
    )

    result = service.run(FIVE_STAGES_QUERY)

    _assert_pipeline_stages(result.answer)
    assert generation.generate_from_chunks.call_count == 1
    assert len(result.steps) == 2
    assert result.steps[1].action.type is AgentActionType.FINISH
    assert result.metadata["finished"] is True
    assert all(step.action.tool_name != DOCUMENT_NAVIGATION_TOOL_NAME for step in result.steps)


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
    assert result.steps[1].action.type is AgentActionType.FINISH


def test_max_steps_one_prevents_navigation_recovery() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="The Core Pipeline begins with Ingest.",
        citations=[],
    )
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    rag = MagicMock()
    rag.retrieve_context.return_value = RetrievalContext(
        query="pipeline",
        chunks=[_fragmented_ingest_chunk()],
    )
    service = AgentService(
        agent=_foundation_agent(),
        tools=ToolRegistry([RAGRetrievalTool(rag), nav_tool]),
        rag_service=generation,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )

    result = service.run(FIVE_STAGES_QUERY)

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
                arguments={"query": FIVE_STAGES_QUERY},
            ),
            observation=AgentObservation(
                tool_name=RAG_RETRIEVAL_TOOL_NAME,
                success=True,
                tool_output=rag_observation_output.model_dump(),
                answer="The Core Pipeline begins with Ingest.",
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
                answer="The Core Pipeline begins with Ingest.",
                metadata={"generated": True, "recovery": True},
            ),
        ),
    ]
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    recovery = maybe_document_navigation_recovery(
        history,
        query=FIVE_STAGES_QUERY,
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
                arguments={"query": FIVE_STAGES_QUERY},
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
                answer="The Core Pipeline begins with Ingest.",
                metadata={"generated": True},
            ),
        )
    ]
    nav_tool = DocumentNavigationTool(
        DocumentNavigationService(make_settings(), MagicMock(spec=VectorStore)),
    )
    recovery = maybe_document_navigation_recovery(
        history,
        query=FIVE_STAGES_QUERY,
        tools=ToolRegistry([RAGRetrievalTool(MagicMock()), nav_tool]),
    )
    assert recovery is None


def test_is_insufficient_rag_answer_detects_refusals() -> None:
    assert is_insufficient_rag_answer("I don't have enough information in the provided context.")
    assert is_insufficient_rag_answer(
        "I could not find relevant information in the knowledge base to answer that question."
    )
    assert not is_insufficient_rag_answer("The pipeline has five stages.")
