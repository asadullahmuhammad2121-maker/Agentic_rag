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
    _reference_label_from_query,
    _select_navigation_anchor,
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
AFTER_PLANNING_QUERY = (
    "According to my uploaded SDLC document, what stage comes immediately after the Planning stage?"
)
AFTER_REQUIREMENTS_QUERY = (
    "According to my uploaded SDLC document, what comes immediately after Requirements Analysis?"
)
BEFORE_TESTING_QUERY = (
    "According to my uploaded SDLC document, what stage comes before Testing?"
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


def _testing_chunk() -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00006",
        text=f"5\n{SDLC_STAGE_TEXT['Testing']}",
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=6,
        chunking_strategy="recursive",
        score=0.90,
    )


def _design_chunk() -> RetrievedChunkOutput:
    return RetrievedChunkOutput(
        chunk_id=f"{SDLC_DOCUMENT_ID}:00004",
        text=f"3\n{SDLC_STAGE_TEXT['Design']}",
        document_id=SDLC_DOCUMENT_ID,
        filename=SDLC_FILENAME,
        file_type="pdf",
        source=SDLC_FILENAME,
        page_number=1,
        section=None,
        chunk_index=4,
        chunking_strategy="recursive",
        score=0.95,
    )


def _rag_chunks_missing_adjacent_successor() -> list[RetrievedChunkOutput]:
    """Simulate production RAG: Planning present, successor section absent."""
    return [_planning_chunk(), _intro_chunk()]


def test_reference_label_extracts_planning_from_after_query() -> None:
    assert _reference_label_from_query(AFTER_PLANNING_QUERY) == "Planning"


def test_reference_label_extracts_multi_word_stage_name() -> None:
    assert _reference_label_from_query(AFTER_REQUIREMENTS_QUERY) == "Requirements Analysis"


def test_reference_label_extracts_testing_from_before_query() -> None:
    assert _reference_label_from_query(BEFORE_TESTING_QUERY) == "Testing"


def test_after_planning_incomplete_when_successor_chunk_missing() -> None:
    retrieval = RAGRetrievalOutput(
        query="sdlc",
        chunks=_rag_chunks_missing_adjacent_successor(),
    )
    assert retrieval_likely_incomplete_for_query(AFTER_PLANNING_QUERY, retrieval)


def test_after_planning_recovery_anchors_planning_chunk() -> None:
    retrieval = RAGRetrievalOutput(
        query="sdlc",
        chunks=_rag_chunks_missing_adjacent_successor(),
    )
    anchor = _select_navigation_anchor(AFTER_PLANNING_QUERY, retrieval.chunks)
    assert anchor.chunk_index == 2
    assert anchor.chunk_id == f"{SDLC_DOCUMENT_ID}:00002"


def test_after_planning_recovery_uses_section_navigation() -> None:
    retrieval = RAGRetrievalOutput(
        query="sdlc",
        chunks=_rag_chunks_missing_adjacent_successor(),
    )
    recovery = maybe_document_navigation_recovery(
        [
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                    arguments={"query": AFTER_PLANNING_QUERY},
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
        query=AFTER_PLANNING_QUERY,
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
    assert recovery.arguments["chunk_id"] == f"{SDLC_DOCUMENT_ID}:00002"
    assert recovery.arguments.get("reference_label") == "Planning"
    assert recovery.arguments.get("direction") == "after"
    assert "window" not in recovery.arguments
    assert "page_number" not in recovery.arguments


def _production_like_sdlc_nav_records() -> dict[tuple[str, int], dict[str, Any]]:
    """Simulate production layout: Planning 38-46, Requirements Analysis 47-51."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for index in range(37):
        records[(SDLC_DOCUMENT_ID, index)] = _payload(
            document_id=SDLC_DOCUMENT_ID,
            chunk_index=index,
            chunk_id=f"{SDLC_DOCUMENT_ID}:{index:05d}",
            text=f"Preamble chunk {index}",
        )
    records[(SDLC_DOCUMENT_ID, 37)] = _payload(
        document_id=SDLC_DOCUMENT_ID,
        chunk_index=37,
        chunk_id=f"{SDLC_DOCUMENT_ID}:00037",
        text="Page 3",
    )
    records[(SDLC_DOCUMENT_ID, 38)] = _payload(
        document_id=SDLC_DOCUMENT_ID,
        chunk_index=38,
        chunk_id=f"{SDLC_DOCUMENT_ID}:00038",
        text="SDLC Reference Document 1. Planning",
    )
    for index in range(39, 47):
        records[(SDLC_DOCUMENT_ID, index)] = _payload(
            document_id=SDLC_DOCUMENT_ID,
            chunk_index=index,
            chunk_id=f"{SDLC_DOCUMENT_ID}:{index:05d}",
            text=f"Planning segment {index}.",
        )
    records[(SDLC_DOCUMENT_ID, 47)] = _payload(
        document_id=SDLC_DOCUMENT_ID,
        chunk_index=47,
        chunk_id=f"{SDLC_DOCUMENT_ID}:00047",
        text="2. Requirements Analysis",
    )
    for index in range(48, 52):
        records[(SDLC_DOCUMENT_ID, index)] = _payload(
            document_id=SDLC_DOCUMENT_ID,
            chunk_index=index,
            chunk_id=f"{SDLC_DOCUMENT_ID}:{index:05d}",
            text=f"Requirements Analysis segment {index}.",
        )
    return records


def _production_like_planning_rag_chunks() -> list[RetrievedChunkOutput]:
    return [
        RetrievedChunkOutput(
            chunk_id=f"{SDLC_DOCUMENT_ID}:00039",
            text="Planning segment 39.",
            document_id=SDLC_DOCUMENT_ID,
            filename=SDLC_FILENAME,
            file_type="pdf",
            source=SDLC_FILENAME,
            page_number=1,
            section=None,
            chunk_index=39,
            chunking_strategy="recursive",
            score=0.90,
        ),
        RetrievedChunkOutput(
            chunk_id=f"{SDLC_DOCUMENT_ID}:00038",
            text="SDLC Reference Document 1. Planning",
            document_id=SDLC_DOCUMENT_ID,
            filename=SDLC_FILENAME,
            file_type="pdf",
            source=SDLC_FILENAME,
            page_number=1,
            section=None,
            chunk_index=38,
            chunking_strategy="recursive",
            score=0.88,
        ),
        RetrievedChunkOutput(
            chunk_id=f"{SDLC_DOCUMENT_ID}:00040",
            text="Planning segment 40.",
            document_id=SDLC_DOCUMENT_ID,
            filename=SDLC_FILENAME,
            file_type="pdf",
            source=SDLC_FILENAME,
            page_number=1,
            section=None,
            chunk_index=40,
            chunking_strategy="recursive",
            score=0.87,
        ),
    ]


def test_production_like_after_planning_merged_context_includes_requirements_section() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="Requirements Analysis follows Planning.",
            citations=[],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=chunk.chunk_index,
                chunking_strategy="recursive",
                score=chunk.score,
            )
            for chunk in _production_like_planning_rag_chunks()
        ],
        nav_records=_production_like_sdlc_nav_records(),
        generation=generation,
    )

    service.run(AFTER_PLANNING_QUERY)

    merged = generation.generate_from_chunks.call_args_list[1].args[1]
    merged_indices = {chunk.chunk_index for chunk in merged}
    merged_text = "\n".join(chunk.text for chunk in merged)
    assert merged_indices.intersection({47, 48, 49, 50, 51})
    assert "Requirements Analysis" in merged_text


def test_after_planning_skips_navigation_when_requirements_already_in_context() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.return_value = RAGResult(
        answer="Requirements Analysis comes after Planning.",
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
            ),
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
            ),
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    result = service.run(AFTER_PLANNING_QUERY)

    assert generation.generate_from_chunks.call_count == 1
    assert all(
        step.action.tool_name != DOCUMENT_NAVIGATION_TOOL_NAME for step in result.steps
    )


def test_after_planning_merged_context_includes_requirements_chunk() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="Requirements Analysis follows Planning.",
            citations=[],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=chunk.chunk_index,
                chunking_strategy="recursive",
                score=chunk.score,
            )
            for chunk in _rag_chunks_missing_adjacent_successor()
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    service.run(AFTER_PLANNING_QUERY)

    merged = generation.generate_from_chunks.call_args_list[1].args[1]
    merged_indices = {chunk.chunk_index for chunk in merged}
    merged_text = "\n".join(chunk.text for chunk in merged)
    assert 3 in merged_indices
    assert "Requirements Analysis" in merged_text
    assert SDLC_STAGE_TEXT["Requirements Analysis"] in merged_text


def test_after_requirements_merged_context_includes_design_chunk() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="Design follows Requirements Analysis.",
            citations=[],
        ),
    ]
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

    service.run(AFTER_REQUIREMENTS_QUERY)

    merged = generation.generate_from_chunks.call_args_list[1].args[1]
    merged_indices = {chunk.chunk_index for chunk in merged}
    assert 4 in merged_indices
    assert SDLC_STAGE_TEXT["Design"] in "\n".join(chunk.text for chunk in merged)


def test_before_testing_merged_context_includes_implementation_chunk() -> None:
    generation = MagicMock()
    generation.generate_from_chunks.side_effect = [
        RAGResult(
            answer="I do not have enough information in the provided context.",
            citations=[],
        ),
        RAGResult(
            answer="Implementation comes before Testing.",
            citations=[],
        ),
    ]
    service = _recovery_service(
        rag_chunks=[
            RetrievedChunk(
                chunk_id=_testing_chunk().chunk_id,
                text=_testing_chunk().text,
                document_id=SDLC_DOCUMENT_ID,
                filename=SDLC_FILENAME,
                file_type="pdf",
                source=SDLC_FILENAME,
                page_number=1,
                section=None,
                chunk_index=6,
                chunking_strategy="recursive",
                score=0.90,
            )
        ],
        nav_records=_fragmented_sdlc_nav_records(),
        generation=generation,
    )

    service.run(BEFORE_TESTING_QUERY)

    merged = generation.generate_from_chunks.call_args_list[1].args[1]
    merged_indices = {chunk.chunk_index for chunk in merged}
    assert 5 in merged_indices
    assert SDLC_STAGE_TEXT["Implementation"] in "\n".join(chunk.text for chunk in merged)


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
