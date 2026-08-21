"""Regression tests for document upload, deduplication, rollback, and delete."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ingestion_service
from app.core.exceptions import (
    DocumentIngestionError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidDocumentError,
    ProviderError,
)
from app.main import create_app
from app.services.ingestion.service import DocumentIngestionService
from app.services.retrieval.keyword.bm25 import BM25KeywordSearch
from app.utils.checksum import sha256_digest
from app.vector_store.base import SearchResult, VectorRecord, VectorStore
from tests.conftest import make_settings
from tests.helpers.pdf_fixtures import (
    build_corrupt_pdf_bytes,
    build_pdf_bytes,
)


class _InMemoryVectorStore(VectorStore):
    """Minimal vector store tracking payloads for lifecycle tests."""

    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.add_calls = 0

    def health_check(self) -> bool:
        return True

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        *,
        distance: str = "Cosine",
    ) -> None:
        return None

    def delete_collection(self, collection_name: str) -> None:
        self.records.clear()

    def add_vectors(self, collection_name: str, records: list[VectorRecord]) -> None:
        self.add_calls += 1
        for record in records:
            self.records[str(record.id)] = record

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: Any = None,
    ) -> list[SearchResult]:
        return []

    def delete(self, collection_name: str, ids: list[str | Any]) -> None:
        for point_id in ids:
            self.records.pop(str(point_id), None)

    def delete_by_payload(self, collection_name: str, conditions: dict[str, Any]) -> None:
        to_delete = [
            point_id
            for point_id, record in self.records.items()
            if _matches_payload(record.payload, conditions)
        ]
        for point_id in to_delete:
            del self.records[point_id]

    def count_by_payload(self, collection_name: str, conditions: dict[str, Any]) -> int:
        return sum(
            1 for record in self.records.values() if _matches_payload(record.payload, conditions)
        )

    def find_by_payload(
        self,
        collection_name: str,
        conditions: dict[str, Any],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        matches = [
            SearchResult(id=point_id, score=1.0, payload=dict(record.payload))
            for point_id, record in self.records.items()
            if _matches_payload(record.payload, conditions)
        ]
        return matches[:limit]

    def ensure_payload_index(
        self,
        collection_name: str,
        field_name: str,
        *,
        field_schema: str = "keyword",
    ) -> None:
        return None


def _matches_payload(payload: dict[str, Any], conditions: dict[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in conditions.items())


@pytest.fixture
def embedding_service() -> MagicMock:
    service = MagicMock()
    service.provider_name = "huggingface"
    service.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] for index, _ in enumerate(texts)]

    service.embed_documents.side_effect = _embed
    return service


@pytest.fixture
def vector_store() -> _InMemoryVectorStore:
    return _InMemoryVectorStore()


@pytest.fixture
def keyword_search(tmp_path: Path) -> BM25KeywordSearch:
    return BM25KeywordSearch(tmp_path / "index.json")


@pytest.fixture
def service(
    vector_store: _InMemoryVectorStore,
    embedding_service: MagicMock,
    keyword_search: BM25KeywordSearch,
) -> DocumentIngestionService:
    settings = make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5)
    return DocumentIngestionService(
        settings,
        vector_store,
        embedding_service,
        keyword_search=keyword_search,
    )


def test_first_upload_succeeds(service: DocumentIngestionService) -> None:
    content = build_pdf_bytes(["Lifecycle upload"])
    result = service.ingest_document(
        filename="first.pdf",
        content=content,
        content_type="application/pdf",
    )

    assert result.document_id
    assert result.checksum == sha256_digest(content)
    assert result.chunks_stored >= 1


def test_duplicate_upload_is_rejected(service: DocumentIngestionService) -> None:
    content = build_pdf_bytes(["Duplicate lifecycle"])
    first = service.ingest_document(
        filename="dup.pdf",
        content=content,
        content_type="application/pdf",
    )

    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.ingest_document(
            filename="dup-copy.pdf",
            content=content,
            content_type="application/pdf",
        )

    assert exc_info.value.details.get("existing_document_id") == first.document_id


def test_parse_failure_then_retry_succeeds(service: DocumentIngestionService) -> None:
    good_content = build_pdf_bytes(["Retry after parse failure"])
    with pytest.raises(InvalidDocumentError):
        service.ingest_document(
            filename="bad.pdf",
            content=build_corrupt_pdf_bytes(),
            content_type="application/pdf",
        )

    result = service.ingest_document(
        filename="good.pdf",
        content=good_content,
        content_type="application/pdf",
    )
    assert result.filename == "good.pdf"


def test_embedding_failure_then_retry_succeeds(
    vector_store: _InMemoryVectorStore,
    keyword_search: BM25KeywordSearch,
) -> None:
    embedding = MagicMock()
    embedding.provider_name = "huggingface"
    embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding.embed_documents.side_effect = [
        ProviderError("upstream failed", provider="huggingface", details={"reason": "api_failure"}),
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
    ]
    service = DocumentIngestionService(
        make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5),
        vector_store,
        embedding,
        keyword_search=keyword_search,
    )
    content = build_pdf_bytes(["Retry after embedding failure"])

    with pytest.raises(ProviderError):
        service.ingest_document(
            filename="embed-fail.pdf",
            content=content,
            content_type="application/pdf",
        )
    assert vector_store.records == {}

    result = service.ingest_document(
        filename="embed-fail.pdf",
        content=content,
        content_type="application/pdf",
    )
    assert result.document_id
    assert vector_store.records


def test_storage_failure_then_retry_succeeds(
    vector_store: _InMemoryVectorStore,
    embedding_service: MagicMock,
    keyword_search: BM25KeywordSearch,
) -> None:
    settings = make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5)
    service = DocumentIngestionService(
        settings,
        vector_store,
        embedding_service,
        keyword_search=keyword_search,
    )
    content = build_pdf_bytes(["Retry after storage failure"])
    calls = {"count": 0}

    original_add = vector_store.add_vectors

    def flaky_add(collection_name: str, records: list[VectorRecord]) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise DocumentIngestionError("simulated vector store failure")
        original_add(collection_name, records)

    vector_store.add_vectors = flaky_add  # type: ignore[method-assign]

    with pytest.raises(DocumentIngestionError):
        service.ingest_document(
            filename="store-fail.pdf",
            content=content,
            content_type="application/pdf",
        )
    assert vector_store.records == {}

    result = service.ingest_document(
        filename="store-fail.pdf",
        content=content,
        content_type="application/pdf",
    )
    assert result.document_id


def test_partial_ingestion_is_rolled_back(service: DocumentIngestionService) -> None:
    content = build_pdf_bytes(["Rollback partial ingestion"])
    with (
        patch(
            "app.services.ingestion.service.IngestedDocument",
            side_effect=RuntimeError("simulated post-store failure"),
        ),
        pytest.raises(RuntimeError, match="simulated post-store failure"),
    ):
        service.ingest_document(
            filename="rollback.pdf",
            content=content,
            content_type="application/pdf",
        )

    assert service._vector_store.count_by_payload(  # type: ignore[attr-defined]
        make_settings().qdrant_collection_name,
        {"checksum": sha256_digest(content)},
    ) == 0


def test_delete_then_reupload_succeeds(service: DocumentIngestionService) -> None:
    content = build_pdf_bytes(["Delete and re-upload"])
    first = service.ingest_document(
        filename="guide.pdf",
        content=content,
        content_type="application/pdf",
    )

    deleted = service.delete_document(first.document_id)
    assert deleted.chunks_deleted >= 1
    assert deleted.document_id == first.document_id

    second = service.ingest_document(
        filename="guide.pdf",
        content=content,
        content_type="application/pdf",
    )
    assert second.document_id != first.document_id
    assert second.checksum == first.checksum


def test_delete_removes_all_document_chunks(
    service: DocumentIngestionService,
    vector_store: _InMemoryVectorStore,
) -> None:
    content = build_pdf_bytes(["Page one", "Page two"])
    ingested = service.ingest_document(
        filename="multi.pdf",
        content=content,
        content_type="application/pdf",
    )

    service.delete_document(ingested.document_id)

    remaining = [
        record
        for record in vector_store.records.values()
        if record.payload.get("document_id") == ingested.document_id
    ]
    assert remaining == []
    assert vector_store.count_by_payload(
        make_settings().qdrant_collection_name,
        {"checksum": ingested.checksum},
    ) == 0


def test_failed_upload_does_not_leave_dedup_record(
    vector_store: _InMemoryVectorStore,
    embedding_service: MagicMock,
) -> None:
    service = DocumentIngestionService(
        make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5),
        vector_store,
        embedding_service,
    )
    content = build_pdf_bytes(["No stale dedup"])

    with pytest.raises(InvalidDocumentError):
        service.ingest_document(
            filename="empty.pdf",
            content=b"",
            content_type="application/pdf",
        )

    checksum = sha256_digest(content)
    collection = make_settings().qdrant_collection_name
    assert vector_store.find_by_payload(collection, {"checksum": checksum}, limit=1) == []

    result = service.ingest_document(
        filename="valid.pdf",
        content=content,
        content_type="application/pdf",
    )
    assert result.checksum == checksum


def test_concurrent_duplicate_uploads_do_not_double_ingest(
    vector_store: _InMemoryVectorStore,
    embedding_service: MagicMock,
) -> None:
    service = DocumentIngestionService(
        make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5),
        vector_store,
        embedding_service,
    )
    content = build_pdf_bytes(["Concurrent duplicate"])
    checksum = sha256_digest(content)
    collection = make_settings().qdrant_collection_name
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[Exception] = []

    def _upload() -> None:
        barrier.wait()
        try:
            result = service.ingest_document(
                filename="same.pdf",
                content=content,
                content_type="application/pdf",
            )
            results.append(result.document_id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_upload) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], DuplicateDocumentError)
    assert vector_store.count_by_payload(collection, {"checksum": checksum}) >= 1


def test_delete_removes_keyword_index_entries(
    service: DocumentIngestionService,
    keyword_search: BM25KeywordSearch,
) -> None:
    content = build_pdf_bytes(["Keyword cleanup"])
    ingested = service.ingest_document(
        filename="keyword.pdf",
        content=content,
        content_type="application/pdf",
    )
    assert keyword_search.search("Keyword cleanup", top_k=1)

    service.delete_document(ingested.document_id)

    assert keyword_search.search("Keyword cleanup", top_k=1) == []


def test_delete_missing_document_raises_not_found(service: DocumentIngestionService) -> None:
    with pytest.raises(DocumentNotFoundError):
        service.delete_document("missing-document-id")


def test_delete_api_returns_deleted_payload() -> None:
    application = create_app()
    mock_ingestion = MagicMock()
    mock_ingestion.delete_document.return_value = MagicMock(
        document_id="doc-1",
        chunks_deleted=3,
        checksum="abc123",
        filename="guide.pdf",
    )
    application.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion

    with TestClient(application) as client:
        response = client.delete("/documents/doc-1")

    application.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["chunks_deleted"] == 3
    assert body["status"] == "deleted"
