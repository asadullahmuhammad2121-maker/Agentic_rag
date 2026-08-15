"""Unit tests for Qdrant payload lookup helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.vector_store.qdrant import QdrantVectorStore
from tests.conftest import make_settings


def test_find_by_payload_maps_results() -> None:
    client = MagicMock()
    point = MagicMock()
    point.id = "p1"
    point.payload = {"checksum": "abc", "document_id": "d1"}
    client.scroll.return_value = ([point], None)

    store = QdrantVectorStore(make_settings(), client=client)
    results = store.find_by_payload("docs", {"checksum": "abc"}, limit=1)

    assert len(results) == 1
    assert results[0].id == "p1"
    assert results[0].payload["document_id"] == "d1"
    client.scroll.assert_called_once()


def test_find_by_payload_empty_conditions() -> None:
    client = MagicMock()
    store = QdrantVectorStore(make_settings(), client=client)
    assert store.find_by_payload("docs", {}) == []
    client.scroll.assert_not_called()


def test_ensure_payload_index_ignores_already_exists() -> None:
    client = MagicMock()
    client.create_payload_index.side_effect = RuntimeError("index already exists")
    store = QdrantVectorStore(make_settings(), client=client)
    store.ensure_payload_index("docs", "checksum")
