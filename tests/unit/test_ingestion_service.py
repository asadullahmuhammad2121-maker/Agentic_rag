"""Unit tests for document ingestion service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import DuplicateDocumentError, InvalidDocumentError, ProviderError
from app.services.ingestion.service import DocumentIngestionService
from app.utils.checksum import sha256_digest
from app.vector_store.base import SearchResult, VectorRecord
from tests.conftest import make_settings
from tests.helpers.pdf_fixtures import (
    build_corrupt_pdf_bytes,
    build_empty_pdf_bytes,
    build_pdf_bytes,
)


@pytest.fixture
def vector_store() -> MagicMock:
    store = MagicMock()
    store.find_by_payload.return_value = []
    store.create_collection.return_value = None
    store.ensure_payload_index.return_value = None
    store.add_vectors.return_value = None
    return store


@pytest.fixture
def embedding_service() -> MagicMock:
    service = MagicMock()
    service.provider_name = "huggingface"
    service.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[float(i), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] for i, _ in enumerate(texts)]

    service.embed_documents.side_effect = _embed
    return service


@pytest.fixture
def service(vector_store: MagicMock, embedding_service: MagicMock) -> DocumentIngestionService:
    settings = make_settings(
        embedding_dimension=8,
        chunk_size=50,
        chunk_overlap=5,
    )
    return DocumentIngestionService(
        settings,
        vector_store,
        embedding_service,
    )


def test_ingest_valid_pdf_chunks_and_embeds(
    service: DocumentIngestionService,
    vector_store: MagicMock,
    embedding_service: MagicMock,
) -> None:
    content = build_pdf_bytes(["Alpha page", "Beta page"])
    result = service.ingest_pdf(
        filename="sample.pdf",
        content=content,
        content_type="application/pdf",
    )

    assert result.filename == "sample.pdf"
    assert result.content_type == "application/pdf"
    assert result.file_size == len(content)
    assert result.checksum == sha256_digest(content)
    assert result.page_count == 2
    assert result.chunks_stored >= 2
    assert result.pages_stored == 2
    assert result.document_id

    embedding_service.embed_documents.assert_called_once()
    vector_store.create_collection.assert_called_once()
    vector_store.add_vectors.assert_called_once()
    _collection, records = vector_store.add_vectors.call_args.args
    assert isinstance(records[0], VectorRecord)
    assert records[0].payload["page_number"] == 1
    assert "chunk_index" in records[0].payload
    assert "chunk_id" in records[0].payload
    assert records[0].payload["embedding_status"] == "ready"
    assert len(records[0].vector) == 8


def test_ingest_preserves_metadata(
    service: DocumentIngestionService,
    vector_store: MagicMock,
) -> None:
    content = build_pdf_bytes(["Only page"])
    result = service.ingest_pdf(
        filename="meta.pdf",
        content=content,
        content_type="application/pdf",
    )
    records: list[VectorRecord] = vector_store.add_vectors.call_args.args[1]
    payload = records[0].payload

    assert payload["document_id"] == result.document_id
    assert payload["filename"] == "meta.pdf"
    assert payload["content_type"] == "application/pdf"
    assert payload["file_type"] == "pdf"
    assert payload["source"] == "meta.pdf"
    assert payload["file_size"] == result.file_size
    assert payload["checksum"] == result.checksum
    assert payload["page_number"] == 1
    assert payload["page_count"] == 1
    assert payload["chunk_index"] == 0
    assert payload["chunk_id"] == f"{result.document_id}:00000"
    assert payload["chunking_strategy"] == "fixed"
    assert payload["embedding_status"] == "ready"
    assert payload["embedding_provider"] == "huggingface"


def test_ingest_stores_real_vectors_in_qdrant(
    service: DocumentIngestionService,
    vector_store: MagicMock,
    embedding_service: MagicMock,
) -> None:
    content = build_pdf_bytes(["Vector storage check"])
    service.ingest_pdf(
        filename="vec.pdf",
        content=content,
        content_type="application/pdf",
    )
    records: list[VectorRecord] = vector_store.add_vectors.call_args.args[1]
    assert embedding_service.embed_documents.called
    assert records[0].vector[0] == 0.0
    assert records[0].payload["text"]
    assert "chunk_checksum" in records[0].payload


def test_ingest_embedding_failure_propagates(
    vector_store: MagicMock,
    embedding_service: MagicMock,
) -> None:
    embedding_service.embed_documents.side_effect = ProviderError(
        "upstream failed",
        provider="huggingface",
        details={"reason": "api_failure"},
    )
    service = DocumentIngestionService(
        make_settings(embedding_dimension=8, chunk_size=50, chunk_overlap=5),
        vector_store,
        embedding_service,
    )
    with pytest.raises(ProviderError):
        service.ingest_pdf(
            filename="fail.pdf",
            content=build_pdf_bytes(["will fail"]),
            content_type="application/pdf",
        )
    vector_store.add_vectors.assert_not_called()


def test_ingest_rejects_invalid_extension(service: DocumentIngestionService) -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_pdf(
            filename="notes.zip",
            content=b"hello",
            content_type="application/zip",
        )
    assert exc_info.value.details.get("reason") == "unsupported_file_type"


def test_ingest_rejects_invalid_content_type(service: DocumentIngestionService) -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_pdf(
            filename="notes.pdf",
            content=build_pdf_bytes(["x"]),
            content_type="image/png",
        )
    assert exc_info.value.details.get("reason") == "invalid_content_type"


def test_ingest_rejects_empty_file(service: DocumentIngestionService) -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_pdf(
            filename="empty.pdf",
            content=build_empty_pdf_bytes(),
            content_type="application/pdf",
        )
    assert exc_info.value.details.get("reason") == "empty_file"


def test_ingest_rejects_corrupted_pdf(service: DocumentIngestionService) -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_pdf(
            filename="bad.pdf",
            content=build_corrupt_pdf_bytes(),
            content_type="application/pdf",
        )
    assert exc_info.value.details.get("reason") == "corrupted_pdf"


def test_ingest_rejects_oversized_file(
    vector_store: MagicMock,
    embedding_service: MagicMock,
) -> None:
    settings = make_settings(max_upload_file_size_mb=1, embedding_dimension=8)
    service = DocumentIngestionService(settings, vector_store, embedding_service)
    content = b"%PDF-1.4\n" + (b"a" * (settings.max_upload_file_size_bytes + 1))
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_pdf(
            filename="big.pdf",
            content=content,
            content_type="application/pdf",
        )
    assert exc_info.value.details.get("reason") == "file_too_large"


def test_ingest_rejects_duplicate(
    service: DocumentIngestionService,
    vector_store: MagicMock,
    embedding_service: MagicMock,
) -> None:
    content = build_pdf_bytes(["Duplicate me"])
    checksum = sha256_digest(content)
    vector_store.find_by_payload.return_value = [
        SearchResult(
            id="existing-point",
            score=1.0,
            payload={"document_id": "existing-doc", "checksum": checksum},
        )
    ]

    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.ingest_pdf(
            filename="dup.pdf",
            content=content,
            content_type="application/pdf",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.details.get("existing_document_id") == "existing-doc"
    assert exc_info.value.details.get("checksum") == checksum
    vector_store.add_vectors.assert_not_called()
    embedding_service.embed_documents.assert_not_called()
