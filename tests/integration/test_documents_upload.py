"""API tests for PDF document upload."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ingestion_service, get_vector_store
from app.core.exceptions import DuplicateDocumentError, InvalidDocumentError
from app.main import create_app
from app.services.ingestion.service import IngestedDocument
from tests.helpers.pdf_fixtures import build_pdf_bytes


@pytest.fixture
def upload_client() -> TestClient:
    application = create_app()
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_pdf.return_value = IngestedDocument(
        document_id="doc-123",
        filename="sample.pdf",
        content_type="application/pdf",
        file_size=100,
        checksum="abc",
        page_count=2,
        pages_stored=2,
        chunks_stored=4,
    )
    application.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
    application.dependency_overrides[get_vector_store] = lambda: MagicMock()
    client = TestClient(application)
    client.mock_ingestion = mock_ingestion  # type: ignore[attr-defined]
    yield client
    application.dependency_overrides.clear()


def test_upload_valid_pdf(upload_client: TestClient) -> None:
    content = build_pdf_bytes(["Hello"])
    response = upload_client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", content, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == "doc-123"
    assert body["filename"] == "sample.pdf"
    assert body["status"] == "ingested"
    assert body["page_count"] == 2
    assert body["chunks_stored"] == 4
    upload_client.mock_ingestion.ingest_pdf.assert_called_once()  # type: ignore[attr-defined]


def test_upload_invalid_document_returns_400() -> None:
    application = create_app()
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_pdf.side_effect = InvalidDocumentError(
        "Only PDF files are supported",
        details={"reason": "invalid_extension"},
    )
    application.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    application.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_document"


def test_upload_duplicate_returns_409() -> None:
    application = create_app()
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_pdf.side_effect = DuplicateDocumentError(
        details={"existing_document_id": "doc-1", "checksum": "xyz"},
    )
    application.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("dup.pdf", build_pdf_bytes(["x"]), "application/pdf")},
        )

    application.dependency_overrides.clear()
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "duplicate_document"
    assert body["details"]["existing_document_id"] == "doc-1"
