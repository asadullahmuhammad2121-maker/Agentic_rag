"""Unit tests for Qdrant vector store (mocked client)."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import QdrantConnectionError, VectorStoreError
from app.vector_store.base import VectorRecord
from app.vector_store.filters import PayloadFilter
from app.utils.ids import normalize_point_id
from app.vector_store.qdrant import QdrantVectorStore
from tests.conftest import make_settings


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantVectorStore:
    return QdrantVectorStore(make_settings(), client=mock_client)


def test_health_check_ok(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    mock_client.get_collections.return_value = MagicMock()
    assert store.health_check() is True
    mock_client.get_collections.assert_called_once()


def test_health_check_failure(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    mock_client.get_collections.side_effect = ConnectionError("refused")
    assert store.health_check() is False


def test_create_collection_when_missing(
    store: QdrantVectorStore,
    mock_client: MagicMock,
) -> None:
    mock_client.collection_exists.return_value = False
    store.create_collection("docs", vector_size=384)
    mock_client.create_collection.assert_called_once()


def test_create_collection_when_exists_skips(
    store: QdrantVectorStore,
    mock_client: MagicMock,
) -> None:
    mock_client.collection_exists.return_value = True
    store.create_collection("docs", vector_size=384)
    mock_client.create_collection.assert_not_called()


def test_create_collection_rejects_bad_distance(store: QdrantVectorStore) -> None:
    with pytest.raises(VectorStoreError, match="Unsupported distance"):
        store.create_collection("docs", vector_size=384, distance="Invalid")


def test_delete_collection(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    store.delete_collection("docs")
    mock_client.delete_collection.assert_called_once_with(collection_name="docs")


def test_add_vectors(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    record = VectorRecord(id=str(uuid4()), vector=[0.1, 0.2], payload={"k": "v"})
    store.add_vectors("docs", [record])
    mock_client.upsert.assert_called_once()
    assert mock_client.upsert.call_args.kwargs["collection_name"] == "docs"


def test_add_vectors_empty_noop(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    store.add_vectors("docs", [])
    mock_client.upsert.assert_not_called()


def test_search_maps_results(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    hit = MagicMock()
    hit.id = "abc"
    hit.score = 0.91
    hit.payload = {"source": "x"}
    response = MagicMock()
    response.points = [hit]
    mock_client.query_points.return_value = response

    results = store.search("docs", [0.1, 0.2], limit=5)
    assert len(results) == 1
    assert results[0].id == "abc"
    assert results[0].score == pytest.approx(0.91)
    assert results[0].payload == {"source": "x"}


def test_search_passes_metadata_filters(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    response = MagicMock()
    response.points = []
    mock_client.query_points.return_value = response

    store.search("docs", [0.1], limit=2, filters=PayloadFilter(exact={"document_id": "doc-1"}))
    kwargs = mock_client.query_points.call_args.kwargs
    assert kwargs["limit"] == 2
    assert kwargs["query_filter"] is not None


def test_delete_vectors(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    store.delete("docs", ["1", "2"])
    mock_client.delete.assert_called_once()


def test_connection_error_on_search(
    store: QdrantVectorStore,
    mock_client: MagicMock,
) -> None:
    mock_client.query_points.side_effect = ConnectionError("connection refused")
    with pytest.raises(QdrantConnectionError):
        store.search("docs", [0.1])


def test_normalize_point_id_accepts_chunk_ids() -> None:
    chunk_id = "fa5eedf3-0802-4bff-b144-a17538dc0533:00001"
    normalized = normalize_point_id(chunk_id)
    assert isinstance(normalized, str)
    assert normalized != chunk_id
    assert normalize_point_id(chunk_id) == normalized


def test_normalize_point_id_preserves_uuid_and_numeric_ids() -> None:
    document_id = str(uuid4())
    assert normalize_point_id(document_id) == document_id
    assert normalize_point_id("42") == 42


def test_add_vectors_normalizes_chunk_ids(store: QdrantVectorStore, mock_client: MagicMock) -> None:
    chunk_id = "doc-1:00001"
    record = VectorRecord(id=chunk_id, vector=[0.1, 0.2], payload={"k": "v"})
    store.add_vectors("docs", [record])
    point = mock_client.upsert.call_args.kwargs["points"][0]
    assert str(point.id) != chunk_id
    assert normalize_point_id(chunk_id) == str(point.id)


def test_vector_store_implements_interface() -> None:
    from app.vector_store.base import VectorStore

    assert issubclass(QdrantVectorStore, VectorStore)
