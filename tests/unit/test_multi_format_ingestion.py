"""Unit tests for multi-format ingestion behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import DuplicateDocumentError, InvalidDocumentError
from app.services.ingestion.service import DocumentIngestionService
from app.utils.checksum import sha256_digest
from app.vector_store.base import SearchResult, VectorRecord
from tests.conftest import make_settings
from tests.helpers.document_fixtures import (
    build_csv_bytes,
    build_json_bytes,
    build_markdown_bytes,
    build_txt_bytes,
)
from tests.helpers.pdf_fixtures import build_pdf_bytes


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


def test_ingest_txt_preserves_metadata(
    service: DocumentIngestionService,
    vector_store: MagicMock,
) -> None:
    content = build_txt_bytes("Plain text document body")
    result = service.ingest_document(
        filename="notes.txt",
        content=content,
        content_type="text/plain",
    )

    assert result.file_type == "txt"
    assert result.source == "notes.txt"
    assert result.document_id
    records: list[VectorRecord] = vector_store.add_vectors.call_args.args[1]
    payload = records[0].payload
    assert payload["file_type"] == "txt"
    assert payload["source"] == "notes.txt"
    assert payload["document_id"] == result.document_id
    assert payload["checksum"] == sha256_digest(content)


def test_ingest_assigns_unique_document_ids(
    service: DocumentIngestionService,
) -> None:
    first = service.ingest_document(
        filename="a.txt",
        content=build_txt_bytes("Document A"),
        content_type="text/plain",
    )
    second = service.ingest_document(
        filename="b.txt",
        content=build_txt_bytes("Document B"),
        content_type="text/plain",
    )
    assert first.document_id != second.document_id


def test_ingest_rejects_duplicate_checksum_per_file(
    service: DocumentIngestionService,
    vector_store: MagicMock,
) -> None:
    content = build_txt_bytes("Duplicate content")
    checksum = sha256_digest(content)
    vector_store.find_by_payload.return_value = [
        SearchResult(
            id="existing-point",
            score=1.0,
            payload={"document_id": "existing-doc", "checksum": checksum},
        )
    ]

    with pytest.raises(DuplicateDocumentError) as exc_info:
        service.ingest_document(
            filename="dup.txt",
            content=content,
            content_type="text/plain",
        )

    assert exc_info.value.details.get("checksum") == checksum
    assert exc_info.value.details.get("existing_document_id") == "existing-doc"


def test_ingest_rejects_unsupported_extension(service: DocumentIngestionService) -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_document(
            filename="archive.zip",
            content=b"binary",
            content_type="application/zip",
        )
    assert exc_info.value.details.get("reason") == "unsupported_file_type"


def test_ingest_multiple_documents(
    service: DocumentIngestionService,
    vector_store: MagicMock,
) -> None:
    uploads = [
        ("one.txt", build_txt_bytes("First"), "text/plain"),
        ("two.md", build_markdown_bytes("# Title\nBody"), "text/markdown"),
        ("three.csv", build_csv_bytes([{"name": "Ada"}]), "text/csv"),
    ]
    results = service.ingest_documents(uploads)
    assert len(results) == 3
    assert len({result.document_id for result in results}) == 3
    assert vector_store.add_vectors.call_count == 3


def test_ingest_pdf_backward_compatible(service: DocumentIngestionService) -> None:
    result = service.ingest_pdf(
        filename="legacy.pdf",
        content=build_pdf_bytes(["Legacy PDF flow"]),
        content_type="application/pdf",
    )
    assert result.file_type == "pdf"
    assert result.chunks_stored >= 1


def test_ingest_json_preserves_section_metadata(
    service: DocumentIngestionService,
    vector_store: MagicMock,
) -> None:
    content = build_json_bytes({"topic": "RAG", "phase": "2A"})
    service.ingest_document(
        filename="meta.json",
        content=content,
        content_type="application/json",
    )
    records: list[VectorRecord] = vector_store.add_vectors.call_args.args[1]
    assert records[0].payload["file_type"] == "json"
    assert records[0].payload["section"] is not None
