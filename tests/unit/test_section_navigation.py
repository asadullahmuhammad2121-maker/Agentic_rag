"""Unit tests for section-level document navigation."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.exceptions import QueryError
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
from app.services.agent.tools.document_navigation import DocumentNavigationTool
from app.services.agent.tools.rag import RAG_RETRIEVAL_TOOL_NAME
from app.services.agent.tools.registry import ToolRegistry
from app.services.retrieval.document_navigation import DocumentNavigationService
from app.services.retrieval.section_detection import (
    build_sections,
    extract_heading_label,
    labels_match,
)
from tests.conftest import make_settings
from tests.unit.test_document_navigation_tool import _IndexedVectorStore, _payload, _service

DOC_ID = "doc-sections"
FILENAME = "guide.pdf"


def _production_like_records() -> dict[tuple[str, int], dict[str, Any]]:
    """Planning spans 38-46; Requirements Analysis spans 47-51."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for index in range(37):
        records[(DOC_ID, index)] = _payload(
            document_id=DOC_ID,
            chunk_index=index,
            chunk_id=f"{DOC_ID}:{index:05d}",
            text=f"Document preamble chunk {index}",
        )
    records[(DOC_ID, 37)] = _payload(
        document_id=DOC_ID,
        chunk_index=37,
        chunk_id=f"{DOC_ID}:00037",
        text="Page 3",
    )
    records[(DOC_ID, 38)] = _payload(
        document_id=DOC_ID,
        chunk_index=38,
        chunk_id=f"{DOC_ID}:00038",
        text="SDLC Reference Document 1. Planning",
    )
    for index in range(39, 47):
        records[(DOC_ID, index)] = _payload(
            document_id=DOC_ID,
            chunk_index=index,
            chunk_id=f"{DOC_ID}:{index:05d}",
            text=f"Planning body segment for chunk {index}.",
        )
    records[(DOC_ID, 47)] = _payload(
        document_id=DOC_ID,
        chunk_index=47,
        chunk_id=f"{DOC_ID}:00047",
        text="2. Requirements Analysis",
    )
    for index in range(48, 52):
        records[(DOC_ID, index)] = _payload(
            document_id=DOC_ID,
            chunk_index=index,
            chunk_id=f"{DOC_ID}:{index:05d}",
            text=f"Requirements Analysis body segment for chunk {index}.",
        )
    records[(DOC_ID, 52)] = _payload(
        document_id=DOC_ID,
        chunk_index=52,
        chunk_id=f"{DOC_ID}:00052",
        text="3. System Design",
    )
    records[(DOC_ID, 53)] = _payload(
        document_id=DOC_ID,
        chunk_index=53,
        chunk_id=f"{DOC_ID}:00053",
        text="System Design body segment.",
    )
    records[(DOC_ID, 54)] = _payload(
        document_id=DOC_ID,
        chunk_index=54,
        chunk_id=f"{DOC_ID}:00054",
        text="4. Implementation",
    )
    for index in range(55, 58):
        records[(DOC_ID, index)] = _payload(
            document_id=DOC_ID,
            chunk_index=index,
            chunk_id=f"{DOC_ID}:{index:05d}",
            text=f"Implementation body segment {index}.",
        )
    records[(DOC_ID, 58)] = _payload(
        document_id=DOC_ID,
        chunk_index=58,
        chunk_id=f"{DOC_ID}:00058",
        text="5. Testing",
    )
    for index in range(59, 63):
        records[(DOC_ID, index)] = _payload(
            document_id=DOC_ID,
            chunk_index=index,
            chunk_id=f"{DOC_ID}:{index:05d}",
            text=f"Testing body segment {index}.",
        )
    return records


def _generic_heading_records() -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    records[(DOC_ID, 0)] = _payload(
        document_id=DOC_ID,
        chunk_index=0,
        chunk_id=f"{DOC_ID}:00000",
        text="Introduction",
    )
    records[(DOC_ID, 1)] = _payload(
        document_id=DOC_ID,
        chunk_index=1,
        chunk_id=f"{DOC_ID}:00001",
        text="This document explains the platform architecture.",
    )
    records[(DOC_ID, 2)] = _payload(
        document_id=DOC_ID,
        chunk_index=2,
        chunk_id=f"{DOC_ID}:00002",
        text="Architecture",
    )
    records[(DOC_ID, 3)] = _payload(
        document_id=DOC_ID,
        chunk_index=3,
        chunk_id=f"{DOC_ID}:00003",
        text="The system uses microservices and an event bus.",
    )
    records[(DOC_ID, 4)] = _payload(
        document_id=DOC_ID,
        chunk_index=4,
        chunk_id=f"{DOC_ID}:00004",
        text="Implementation",
    )
    records[(DOC_ID, 5)] = _payload(
        document_id=DOC_ID,
        chunk_index=5,
        chunk_id=f"{DOC_ID}:00005",
        text="Services are deployed with containers.",
    )
    records[(DOC_ID, 6)] = _payload(
        document_id=DOC_ID,
        chunk_index=6,
        chunk_id=f"{DOC_ID}:00006",
        text="Conclusion",
    )
    return records


def test_extract_heading_label_supports_numbered_and_generic_titles() -> None:
    assert extract_heading_label("2. Requirements Analysis") == "Requirements Analysis"
    assert extract_heading_label("Introduction") == "Introduction"
    assert extract_heading_label("Architecture") == "Architecture"
    assert extract_heading_label("Planning body without heading") is None


def test_build_sections_groups_multi_chunk_sections() -> None:
    records = _production_like_records()
    sections = build_sections(
        [(index, records[(DOC_ID, index)]["text"]) for index in range(38, 53)],
    )
    planning = next(section for section in sections if section.label and labels_match(section.label, "Planning"))
    requirements = next(
        section for section in sections if section.label and labels_match(section.label, "Requirements Analysis")
    )
    assert planning.start_index == 38
    assert planning.end_index == 46
    assert requirements.start_index == 47
    assert requirements.end_index == 51


def test_navigate_after_planning_returns_requirements_section_chunks() -> None:
    records = _production_like_records()
    service = _service(records)
    result = service.navigate(
        document_id=DOC_ID,
        chunk_id=f"{DOC_ID}:00039",
        reference_label="Planning",
        direction="after",
    )
    assert [chunk.chunk_index for chunk in result.chunks] == [47, 48, 49, 50, 51]
    assert all("Requirements Analysis" in chunk.text or "requirements" in chunk.text.casefold() for chunk in result.chunks[:2])


def test_navigate_after_requirements_returns_design_section() -> None:
    records = _production_like_records()
    service = _service(records)
    result = service.navigate(
        document_id=DOC_ID,
        chunk_id=f"{DOC_ID}:00048",
        reference_label="Requirements Analysis",
        direction="after",
    )
    assert result.chunks
    assert result.chunks[0].chunk_index == 52
    assert "System Design" in result.chunks[0].text


def test_navigate_before_testing_returns_implementation_section() -> None:
    records = _production_like_records()
    service = _service(records)
    result = service.navigate(
        document_id=DOC_ID,
        chunk_id=f"{DOC_ID}:00059",
        reference_label="Testing",
        direction="before",
    )
    assert result.chunks
    assert min(chunk.chunk_index for chunk in result.chunks) == 54
    assert max(chunk.chunk_index for chunk in result.chunks) == 57
    assert "Implementation" in result.chunks[0].text


def test_navigate_after_generic_heading_returns_next_section() -> None:
    records = _generic_heading_records()
    service = _service(records)
    result = service.navigate(
        document_id=DOC_ID,
        chunk_id=f"{DOC_ID}:00001",
        reference_label="Introduction",
        direction="after",
    )
    assert [chunk.chunk_index for chunk in result.chunks] == [2, 3]
    assert result.chunks[0].text == "Architecture"


def test_navigate_missing_next_section_returns_empty() -> None:
    records = _generic_heading_records()
    service = _service(records)
    result = service.navigate(
        document_id=DOC_ID,
        chunk_id=f"{DOC_ID}:00006",
        reference_label="Conclusion",
        direction="after",
    )
    assert result.chunks == []


def test_navigate_enforces_same_document_scope() -> None:
    records = {
        ("doc-a", 0): _payload(
            document_id="doc-a",
            chunk_index=0,
            chunk_id="doc-a:00000",
            text="1. Alpha",
        ),
        ("doc-a", 1): _payload(
            document_id="doc-a",
            chunk_index=1,
            chunk_id="doc-a:00001",
            text="Alpha body segment.",
        ),
        ("doc-a", 2): _payload(
            document_id="doc-a",
            chunk_index=2,
            chunk_id="doc-a:00002",
            text="2. Beta",
        ),
        ("doc-b", 2): _payload(
            document_id="doc-b",
            chunk_index=2,
            chunk_id="doc-b:00002",
            text="2. Foreign",
        ),
    }
    service = _service(records)
    result = service.navigate(
        document_id="doc-a",
        chunk_id="doc-a:00000",
        reference_label="Alpha",
        direction="after",
    )
    assert [chunk.chunk_index for chunk in result.chunks] == [2]
    assert all(chunk.document_id == "doc-a" for chunk in result.chunks)


def test_recovery_skips_navigation_when_successor_section_already_in_context() -> None:
    query = "What section follows Introduction?"
    retrieval = RAGRetrievalOutput(
        query="guide",
        chunks=[
            RetrievedChunkOutput(
                chunk_id=f"{DOC_ID}:00000",
                text="Introduction",
                document_id=DOC_ID,
                filename=FILENAME,
                file_type="pdf",
                source=FILENAME,
                page_number=1,
                section=None,
                chunk_index=0,
                chunking_strategy="recursive",
                score=0.9,
            ),
            RetrievedChunkOutput(
                chunk_id=f"{DOC_ID}:00001",
                text="This document explains the platform architecture.",
                document_id=DOC_ID,
                filename=FILENAME,
                file_type="pdf",
                source=FILENAME,
                page_number=1,
                section=None,
                chunk_index=1,
                chunking_strategy="recursive",
                score=0.89,
            ),
            RetrievedChunkOutput(
                chunk_id=f"{DOC_ID}:00002",
                text="Architecture",
                document_id=DOC_ID,
                filename=FILENAME,
                file_type="pdf",
                source=FILENAME,
                page_number=1,
                section=None,
                chunk_index=2,
                chunking_strategy="recursive",
                score=0.88,
            ),
        ],
    )
    assert not retrieval_likely_incomplete_for_query(query, retrieval)
    recovery = maybe_document_navigation_recovery(
        [
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                    arguments={"query": query},
                ),
                observation=AgentObservation(
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    success=True,
                    tool_output=retrieval.model_dump(),
                    answer="Architecture follows Introduction.",
                    metadata={"generated": True},
                ),
            )
        ],
        query=query,
        tools=ToolRegistry(
            [
                DocumentNavigationTool(
                    DocumentNavigationService(make_settings(), _IndexedVectorStore({})),
                ),
            ]
        ),
    )
    assert recovery is None


def test_recovery_after_planning_uses_section_navigation_arguments() -> None:
    query = (
        "According to my uploaded document, what stage comes immediately after the Planning stage?"
    )
    retrieval = RAGRetrievalOutput(
        query="doc",
        chunks=[
            RetrievedChunkOutput(
                chunk_id=f"{DOC_ID}:00039",
                text="Planning body segment.",
                document_id=DOC_ID,
                filename=FILENAME,
                file_type="pdf",
                source=FILENAME,
                page_number=1,
                section=None,
                chunk_index=39,
                chunking_strategy="recursive",
                score=0.9,
            ),
            RetrievedChunkOutput(
                chunk_id=f"{DOC_ID}:00038",
                text="SDLC Reference Document 1. Planning",
                document_id=DOC_ID,
                filename=FILENAME,
                file_type="pdf",
                source=FILENAME,
                page_number=1,
                section=None,
                chunk_index=38,
                chunking_strategy="recursive",
                score=0.88,
            ),
        ],
    )
    recovery = maybe_document_navigation_recovery(
        [
            AgentStep(
                action=AgentAction(
                    type=AgentActionType.CALL_TOOL,
                    tool_name=RAG_RETRIEVAL_TOOL_NAME,
                    tool_names=[RAG_RETRIEVAL_TOOL_NAME],
                    arguments={"query": query},
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
        query=query,
        tools=ToolRegistry(
            [
                DocumentNavigationTool(
                    DocumentNavigationService(make_settings(), _IndexedVectorStore({})),
                ),
            ]
        ),
    )
    assert recovery is not None
    assert recovery.arguments["chunk_id"] == f"{DOC_ID}:00039"
    assert recovery.arguments["reference_label"] == "Planning"
    assert recovery.arguments["direction"] == "after"
    assert "window" not in recovery.arguments


def test_document_navigation_tool_requires_section_pair() -> None:
    tool = DocumentNavigationTool(_service(_production_like_records()))
    with pytest.raises(QueryError):
        tool.run(
            {
                "document_id": DOC_ID,
                "chunk_id": f"{DOC_ID}:00039",
                "reference_label": "Planning",
            }
        )
