"""Regression tests for SDLC-style list/stage retrieval and navigation recovery."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.agent.models import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentStep,
    RAGRetrievalOutput,
    RetrievedChunkOutput,
)
from app.services.agent.recovery.navigation import (
    maybe_document_navigation_recovery,
    retrieval_likely_incomplete_for_query,
)
from app.services.agent.tools.document_navigation import (
    DOCUMENT_NAVIGATION_TOOL_NAME,
    DocumentNavigationTool,
)
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.rag.service import RAGResult
from app.services.retrieval.document_navigation import DocumentNavigationService
from app.services.retrieval.service import RetrievedChunk
from tests.conftest import make_settings
from tests.unit.test_agent_navigation_recovery import _recovery_service

SDLC_DOCUMENT_ID = "sdlc-doc"
SDLC_FILENAME = "SDLC_Document.pdf"

SDLC_STAGES = (
    "Planning",
    "Requirements Analysis",
    "Design",
    "Implementation",
    "Testing",
    "Deployment",
    "Maintenance",
)

SDLC_STAGE_TEXT = {
    "Planning": "Planning: Define project scope, objectives, timelines, and resource allocation.",
    "Requirements Analysis": (
        "Requirements Analysis: Gather and document functional and non-functional requirements."
    ),
    "Design": "Design: Create system architecture, data models, and interface specifications.",
    "Implementation": "Implementation: Write, review, and integrate source code.",
    "Testing": "Testing: Execute unit, integration, and system tests to validate quality.",
    "Deployment": "Deployment: Release the application to production environments.",
    "Maintenance": "Maintenance: Monitor, fix defects, and apply enhancements after release.",
}

FIRST_STAGE_QUERY = (
    "According to my uploaded SDLC document, what is the first stage of the "
    "Software Development Lifecycle?"
)
SEVEN_STAGES_QUERY = (
    "According to my uploaded SDLC document, what are the seven stages of the "
    "Software Development Lifecycle?"
)
LIST_SEVEN_STAGES_QUERY = (
    "According to my uploaded SDLC document, list all seven stages of the "
    "Software Development Lifecycle."
)
PLANNING_PURPOSE_QUERY = (
    "According to my uploaded SDLC document, what is the purpose of the Planning stage?"
)
REQUIREMENTS_QUERY = (
    "According to my uploaded SDLC document, what happens during Requirements Analysis?"
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
        "filename": SDLC_FILENAME,
        "file_type": "pdf",
        "source": SDLC_FILENAME,
        "chunking_strategy": "recursive",
    }


def _fragmented_sdlc_nav_records() -> dict[tuple[str, int], dict[str, Any]]:
    """Simulate fragmented PDF-style SDLC chunks (one stage per chunk)."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    records[(SDLC_DOCUMENT_ID, 0)] = _payload(
        document_id=SDLC_DOCUMENT_ID,
        chunk_index=0,
        chunk_id=f"{SDLC_DOCUMENT_ID}:00000",
        text="The Software Development Lifecycle (SDLC) defines how teams build software.",
    )
    records[(SDLC_DOCUMENT_ID, 1)] = _payload(
        document_id=SDLC_DOCUMENT_ID,
        chunk_index=1,
        chunk_id=f"{SDLC_DOCUMENT_ID}:00001",
        text="Software Development Lifecycle",
    )
    for index, stage in enumerate(SDLC_STAGES, start=2):
        records[(SDLC_DOCUMENT_ID, index)] = _payload(
            document_id=SDLC_DOCUMENT_ID,
            chunk_index=index,
            chunk_id=f"{SDLC_DOCUMENT_ID}:{index:05d}",
            text=f"{index - 1}\n{SDLC_STAGE_TEXT[stage]}",
        )
    return records


def _intro_chunk() -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00000",
        text="The Software Development Lifecycle (SDLC) defines how teams build software.",
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=0,
        chunking_strategy="recursive",
        score=0.92,
    )


def _planning_chunk() -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00002",
        text=f"1\n{SDLC_STAGE_TEXT['Planning']}",
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=2,
        chunking_strategy="recursive",
        score=0.88,
    )


def _requirements_chunk() -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00003",
        text=f"2\n{SDLC_STAGE_TEXT['Requirements Analysis']}",
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=3,
        chunking_strategy="recursive",
        score=0.91,
    )


def _complete_sdlc_list_chunk() -> RetrievedChunkOutput:
    lines = ["Software Development Lifecycle"]
    for index, stage in enumerate(SDLC_STAGES, start=1):
        lines.append(str(index))
        lines.append(SDLC_STAGE_TEXT[stage])
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00001",
        text="\n".join(lines),
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=1,
        chunking_strategy="recursive",
        score=0.95,
    )


def test_seven_stages_query_detected_as_incomplete_with_intro_only() -> None:
    retrieval = RAGRetrievalOutput(query="sdlc", chunks=[_intro_chunk()])
    assert retrieval_likely_incomplete_for_query(SEVEN_STAGES_QUERY, retrieval)


def test_first_stage_query_detected_as_incomplete_with_intro_only() -> None:
    retrieval = RAGRetrievalOutput(query="sdlc", chunks=[_intro_chunk()])
    assert retrieval_likely_incomplete_for_query(FIRST_STAGE_QUERY, retrieval)


def test_planning_purpose_query_not_incomplete_with_planning_chunk() -> None:
    retrieval = RAGRetrievalOutput(query="sdlc", chunks=[_planning_chunk()])
    assert not retrieval_likely_incomplete_for_query(PLANNING_PURPOSE_QUERY, retrieval)


def test_requirements_query_not_incomplete_with_requirements_chunk() -> None:
    retrieval = RAGRetrievalOutput(query="sdlc", chunks=[_requirements_chunk()])
    assert not retrieval_likely_incomplete_for_query(REQUIREMENTS_QUERY, retrieval)


def test_seven_stages_recovery_uses_page_navigation() -> None:
    retrieval = RAGRetrievalOutput(query="sdlc", chunks=[_intro_chunk()])
    recovery = maybe_document_navigation_recovery(
        [
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                    arguments={"query": SEVEN_STAGES_QUERY},
                ),
                observation=AgentObservation(
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    success=True,
                    tool_output=retrieval.model_dump(),
                    answer="I do not have enough information in the provided context.",
                    metadata={"generated": True},
                ),
            )
        ],
        query=SEVEN_STAGES_QUERY,
        tools=ToolRegistry(
            [
                MagicMock(),
                DocumentNavigationTool(
                    DocumentNavigationService(make_settings(), MagicMock()),
                ),
            ]
        ),
    )
    assert recovery is not None
    assert recovery.arguments.get("page_number") == 1
    assert recovery.arguments.get("limit") == 20
    assert "chunk_id" not in recovery.arguments


def test_fragmented_seven_stages_query_recovers_via_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer=(
                "The seven stages are Planning, Requirements Analysis, Design, "
                "Implementation, Testing, Deployment, and Maintenance."
            ),
            citations=[],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_intro_chunk().chunk_id,
                text=_intro_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=0,
                chunking_strategy="recursive",
                score=0.92,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(SEVEN_STAGES_QUERY)

    assert "Planning" in result.answer
    assert "Maintenance" in result.answer
    assert generation.generate_from_chunks.call_count == 2
    merged = generation.generate_from_chunks.call_args_list[1].args[1]
    merged_text = "\n".join(chunk.text for chunk in merged)
    for stage in SDLC_STAGES:
        assert stage in merged_text or SDLC_STAGE_TEXT[stage] in merged_text


def test_fragmented_first_stage_query_recovers_via_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="The first stage is Planning.",
            citations=[],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_intro_chunk().chunk_id,
                text=_intro_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=0,
                chunking_strategy="recursive",
                score=0.92,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(FIRST_STAGE_QUERY)

    assert "Planning" in result.answer
    assert generation.generate_from_chunks.call_count == 2


def test_complete_sdlc_list_skips_navigation() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer=(
            "The seven stages are Planning, Requirements Analysis, Design, "
            "Implementation, Testing, Deployment, and Maintenance."
        ),
        citations=[],
    )
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_complete_sdlc_list_chunk().chunk_id,
                text=_complete_sdlc_list_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=1,
                chunking_strategy="recursive",
                score=0.95,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(LIST_SEVEN_STAGES_QUERY)

    assert "Planning" in result.answer
    assert generation.generate_from_chunks.call_count == 1
    assert all(
        step.action.tool_name != DOCUMENT_NAVIGATION_TOOL_NAME for step in result.steps
    )


def test_planning_purpose_single_stage_still_works() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Planning defines project scope, objectives, timelines, and resource allocation.",
        citations=[],
    )
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_planning_chunk().chunk_id,
                text=_planning_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=2,
                chunking_strategy="recursive",
                score=0.88,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(PLANNING_PURPOSE_QUERY)

    assert "scope" in result.answer.casefold()
    assert generation.generate_from_chunks.call_count == 1


def test_requirements_analysis_single_stage_still_works() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Requirements Analysis gathers and documents stakeholder requirements.",
        citations=[],
    )
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_requirements_chunk().chunk_id,
                text=_requirements_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=3,
                chunking_strategy="recursive",
                score=0.91,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(REQUIREMENTS_QUERY)

    assert "requirements" in result.answer.casefold()
    assert generation.generate_from_chunks.call_count == 1
