"""Unit tests for vector-store payload filter building."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.vector_store.filters import PayloadFilter
from app.vector_store.qdrant import QdrantVectorStore, build_qdrant_filter
from tests.conftest import make_settings


def test_build_qdrant_filter_none_when_empty() -> None:
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter(PayloadFilter()) is None


def test_build_qdrant_filter_exact_match() -> None:
    payload_filter = PayloadFilter(exact={"document_id": "doc-1"})
    qdrant_filter = build_qdrant_filter(payload_filter)
    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 1


def test_build_qdrant_filter_any_match() -> None:
    payload_filter = PayloadFilter(any_of={"file_type": ("pdf", "txt")})
    qdrant_filter = build_qdrant_filter(payload_filter)
    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 1


def test_search_passes_structured_filters() -> None:
    client = MagicMock()
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response
    store = QdrantVectorStore(make_settings(), client=client)

    payload_filter = PayloadFilter(
        exact={"document_id": "doc-1"},
        any_of={"file_type": ("pdf", "txt")},
    )
    store.search("docs", [0.1], limit=2, filters=payload_filter)
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["limit"] == 2
    assert kwargs["query_filter"] is not None
