"""Unit tests for document navigation service and tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import QueryError
from app.services.agent.models import DocumentNavigationOutput
from app.services.agent.service import AgentService
from app.services.agent.tools.converters import tool_result_to_observation
from app.services.agent.tools.document_navigation import (
    DOCUMENT_NAVIGATION_TOOL_NAME,
    DocumentNavigationTool,
)
from app.services.agent.tools.registry import ToolRegistry
from app.services.rag.service import Citation, RAGResult
from app.services.retrieval.document_navigation import DocumentNavigationService
from app.vector_store.base import SearchResult, VectorStore
from tests.conftest import make_settings


def _payload(
    *,
    document_id: str,
    chunk_index: int,
    chunk_id: str,
    text: str,
    page_number: int = 1,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "page_number": page_number,
        "text": text,
        "filename": "guide.pdf",
        "file_type": "pdf",
        "source": "guide.pdf",
        "chunking_strategy": "recursive",
    }


class _IndexedVectorStore:
    """In-memory vector store keyed by document_id + chunk_index."""

    def __init__(self, records: dict[tuple[str, int], dict[str, Any]]) -> None:
        self._records = records
        self.calls: list[dict[str, Any]] = []

    def find_by_payload(
        self,
        collection_name: str,
        conditions: dict[str, Any],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        self.calls.append({"collection": collection_name, "conditions": conditions, "limit": limit})
        document_id = conditions.get("document_id")
        chunk_index = conditions.get("chunk_index")
        chunk_id = conditions.get("chunk_id")
        page_number = conditions.get("page_number")

        if chunk_id is not None and document_id is not None:
            for (doc_id, index), payload in self._records.items():
                if doc_id == document_id and payload["chunk_id"] == chunk_id:
                    return [_result(doc_id, index, payload)]
            return []

        if chunk_index is not None and document_id is not None:
            payload = self._records.get((str(document_id), int(chunk_index)))
            if payload is None:
                return []
            return [_result(str(document_id), int(chunk_index), payload)]

        if page_number is not None and document_id is not None:
            hits = [
                _result(doc_id, index, payload)
                for (doc_id, index), payload in self._records.items()
                if doc_id == document_id and payload.get("page_number") == page_number
            ]
            hits.sort(key=lambda item: item.payload["chunk_index"])
            return hits[:limit]

        return []


def _result(document_id: str, chunk_index: int, payload: dict[str, Any]) -> SearchResult:
    return SearchResult(
        id=payload["chunk_id"],
        score=1.0,
        payload=payload,
    )


def _service(records: dict[tuple[str, int], dict[str, Any]]) -> DocumentNavigationService:
    settings = make_settings()
    return DocumentNavigationService(settings, _IndexedVectorStore(records))  # type: ignore[arg-type]


def test_navigation_returns_neighbors_in_chunk_order() -> None:
    records = {
        ("doc-a", 15): _payload(document_id="doc-a", chunk_index=15, chunk_id="doc-a:00015", text="before"),
        ("doc-a", 16): _payload(document_id="doc-a", chunk_index=16, chunk_id="doc-a:00016", text="anchor"),
        ("doc-a", 17): _payload(document_id="doc-a", chunk_index=17, chunk_id="doc-a:00017", text="after"),
    }
    service = _service(records)
    result = service.navigate(document_id="doc-a", chunk_id="doc-a:00016", window=1)

    assert [chunk.chunk_index for chunk in result.chunks] == [15, 16, 17]
    assert [chunk.text for chunk in result.chunks] == ["before", "anchor", "after"]
    assert result.anchor_chunk_index == 16


def test_navigation_never_returns_cross_document_chunks() -> None:
    records = {
        ("doc-a", 16): _payload(document_id="doc-a", chunk_index=16, chunk_id="doc-a:00016", text="anchor"),
        ("doc-b", 15): _payload(document_id="doc-b", chunk_index=15, chunk_id="doc-b:00015", text="other doc"),
        ("doc-b", 17): _payload(document_id="doc-b", chunk_index=17, chunk_id="doc-b:00017", text="other doc 2"),
    }
    service = _service(records)
    result = service.navigate(document_id="doc-a", chunk_id="doc-a:00016", window=2)

    assert len(result.chunks) == 1
    assert result.chunks[0].document_id == "doc-a"


def test_navigation_respects_requested_limit() -> None:
    records = {
        ("doc-a", index): _payload(
            document_id="doc-a",
            chunk_index=index,
            chunk_id=f"doc-a:{index:05d}",
            text=f"chunk {index}",
        )
        for index in range(10, 21)
    }
    service = _service(records)
    result = service.navigate(document_id="doc-a", chunk_index=15, window=5, limit=3)

    assert len(result.chunks) == 3
    assert result.chunks[1].chunk_index == 15


def test_navigation_by_page_number_returns_page_chunks() -> None:
    records = {
        ("doc-a", 3): _payload(
            document_id="doc-a",
            chunk_index=3,
            chunk_id="doc-a:00003",
            text="page two a",
            page_number=2,
        ),
        ("doc-a", 4): _payload(
            document_id="doc-a",
            chunk_index=4,
            chunk_id="doc-a:00004",
            text="page two b",
            page_number=2,
        ),
        ("doc-a", 5): _payload(
            document_id="doc-a",
            chunk_index=5,
            chunk_id="doc-a:00005",
            text="page one",
            page_number=1,
        ),
    }
    service = _service(records)
    result = service.navigate(document_id="doc-a", page_number=2, limit=10)

    assert [chunk.chunk_index for chunk in result.chunks] == [3, 4]
    assert all(chunk.page_number == 2 for chunk in result.chunks)


def test_navigation_missing_anchor_returns_empty_result() -> None:
    service = _service({})
    result = service.navigate(document_id="doc-a", chunk_id="missing:00001")

    assert result.chunks == []
    assert result.anchor_chunk_id == "missing:00001"


def test_navigation_without_neighbors_returns_empty_result() -> None:
    records = {
        ("doc-a", 16): _payload(document_id="doc-a", chunk_index=16, chunk_id="doc-a:00016", text="only"),
    }
    service = _service(records)
    result = service.navigate(document_id="doc-a", chunk_id="doc-a:00016", window=1)

    assert len(result.chunks) == 1
    assert result.chunks[0].text == "only"


def test_document_navigation_tool_registered() -> None:
    settings = make_settings()
    store = MagicMock(spec=VectorStore)
    tool = DocumentNavigationTool(DocumentNavigationService(settings, store))
    registry = ToolRegistry([tool])
    assert DOCUMENT_NAVIGATION_TOOL_NAME in registry


def test_document_navigation_tool_run_success() -> None:
    records = {
        ("doc-a", 16): _payload(document_id="doc-a", chunk_index=16, chunk_id="doc-a:00016", text="anchor"),
        ("doc-a", 17): _payload(document_id="doc-a", chunk_index=17, chunk_id="doc-a:00017", text="next"),
    }
    tool = DocumentNavigationTool(_service(records))
    result = tool.run({"document_id": "doc-a", "chunk_id": "doc-a:00016", "window": 1})

    assert result.success is True
    assert isinstance(result.output, DocumentNavigationOutput)
    assert result.output.result_count == 2
    observation = tool_result_to_observation(DOCUMENT_NAVIGATION_TOOL_NAME, result)
    assert observation.success is True
    assert observation.metadata["result_count"] == 2


def test_document_navigation_tool_requires_anchor() -> None:
    tool = DocumentNavigationTool(_service({}))
    with pytest.raises(QueryError) as exc_info:
        tool.run({"document_id": "doc-a"})
    assert exc_info.value.details.get("reason") == "invalid_tool_input"


def test_agent_service_executes_document_navigation_tool() -> None:
    from app.services.agent.base import Agent
    from app.services.agent.models import AgentAction, AgentActionType

    records = {
        ("doc-a", 16): _payload(document_id="doc-a", chunk_index=16, chunk_id="doc-a:00016", text="anchor"),
        ("doc-a", 17): _payload(document_id="doc-a", chunk_index=17, chunk_id="doc-a:00017", text="neighbor"),
    }
    navigation_tool = DocumentNavigationTool(_service(records))
    registry = ToolRegistry([navigation_tool])

    class _NavAgent(Agent):
        def decide(self, request, *, tools, history):
            return AgentAction(
                type=AgentActionType.CALL_TOOL,
                tool_name=DOCUMENT_NAVIGATION_TOOL_NAME,
                tool_names=[DOCUMENT_NAVIGATION_TOOL_NAME],
                arguments={
                    "document_id": "doc-a",
                    "chunk_id": "doc-a:00016",
                    "window": 1,
                },
            )

    rag_service = MagicMock()
    rag_service.generate_from_chunks.return_value = RAGResult(
        answer="The nearby context includes anchor and neighbor.",
        citations=[
            Citation(
                document_id="doc-a",
                filename="guide.pdf",
                file_type="pdf",
                source="guide.pdf",
                page_number=1,
                section=None,
                chunk_index=16,
                chunk_id="doc-a:00016",
                score=1.0,
                label="S1",
            )
        ],
    )

    service = AgentService(
        agent=_NavAgent(),
        tools=registry,
        rag_service=rag_service,
        web_answer_generator=MagicMock(),
        max_steps=1,
    )
    result = service.run("What comes after the anchor chunk?")

    assert DOCUMENT_NAVIGATION_TOOL_NAME in result.tool_used
    assert "nearby context" in result.answer
    rag_service.generate_from_chunks.assert_called_once()
    chunks = rag_service.generate_from_chunks.call_args.args[1]
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 16
