"""Hardening tests for security, edge cases, failures, and feature compatibility."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ingestion_service, get_vector_store
from app.core.exceptions import InvalidDocumentError
from app.core.logging import SensitiveDataFilter, StructuredFormatter
from app.main import create_app
from app.services.context_optimization.service import ContextOptimizationService
from app.services.query_transformation.service import TransformedQuery
from app.services.rag.service import RAGService
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.multi_query import MultiQueryRetrievalService
from app.services.retrieval.service import RetrievedChunk
from app.utils.filenames import sanitize_upload_filename
from tests.conftest import make_settings
from tests.helpers.document_fixtures import (
    build_csv_bytes,
    build_docx_bytes,
    build_json_bytes,
    build_markdown_bytes,
    build_txt_bytes,
)
from tests.helpers.pdf_fixtures import build_empty_pdf_bytes, build_pdf_bytes


def _chunk(**overrides: object) -> RetrievedChunk:
    base = {
        "chunk_id": "doc-1:00000",
        "text": "alpha beta gamma delta",
        "document_id": "doc-1",
        "filename": "a.pdf",
        "file_type": "pdf",
        "source": "a.pdf",
        "page_number": 1,
        "section": "Intro",
        "chunk_index": 0,
        "chunking_strategy": "fixed",
        "score": 0.9,
    }
    base.update(overrides)
    return RetrievedChunk(**base)  # type: ignore[arg-type]


# --- Security ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("nested/path/report.pdf", "report.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
    ],
)
def test_sanitize_upload_filename_strips_path_components(raw: str, expected: str) -> None:
    assert sanitize_upload_filename(raw) == expected


def test_sanitize_upload_filename_rejects_traversal_only() -> None:
    with pytest.raises(InvalidDocumentError) as exc_info:
        sanitize_upload_filename("../")
    assert exc_info.value.details.get("reason") == "invalid_filename"


def test_sensitive_filter_redacts_secret_keys_in_formatted_output() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="health_check",
        args=(),
        exc_info=None,
    )
    record.groq_api_key = "super-secret"  # type: ignore[attr-defined]
    record.operation = "test"  # type: ignore[attr-defined]

    filt = SensitiveDataFilter()
    filt.filter(record)
    formatted = StructuredFormatter().format(record)

    assert "super-secret" not in formatted
    assert "[REDACTED]" in formatted


def test_upload_route_rejects_batch_limit() -> None:
    application = create_app()
    application.dependency_overrides[get_ingestion_service] = lambda: MagicMock()
    application.dependency_overrides[get_vector_store] = lambda: MagicMock()

    with TestClient(application, raise_server_exceptions=False) as client:
        files = [
            ("files", (f"doc{i}.txt", b"hello", "text/plain"))
            for i in range(21)
        ]
        response = client.post("/documents/upload", files=files)

    application.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json()["details"]["reason"] == "batch_limit_exceeded"


def test_upload_route_sanitizes_filename_before_ingest() -> None:
    application = create_app()
    mock_ingestion = MagicMock()
    mock_ingestion.ingest_document.return_value = MagicMock(
        document_id="d",
        filename="evil.pdf",
        content_type="application/pdf",
        file_type="pdf",
        file_size=1,
        checksum="x",
        source="evil.pdf",
        page_count=1,
        pages_stored=1,
        chunks_stored=1,
    )
    application.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
    application.dependency_overrides[get_vector_store] = lambda: MagicMock()

    with TestClient(application) as client:
        response = client.post(
            "/documents/upload",
            files={"file": ("../../evil.pdf", build_pdf_bytes(["x"]), "application/pdf")},
        )

    application.dependency_overrides.clear()
    assert response.status_code == 201
    assert mock_ingestion.ingest_document.call_args.kwargs["filename"] == "evil.pdf"


# --- Edge cases ---


@pytest.mark.parametrize(
    ("filename", "content", "content_type"),
    [
        ("doc.pdf", build_pdf_bytes(["PDF body"]), "application/pdf"),
        ("doc.docx", build_docx_bytes([("DOCX body", None)]), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("doc.txt", build_txt_bytes("TXT body"), "text/plain"),
        ("doc.md", build_markdown_bytes("# Title\nMarkdown body"), "text/markdown"),
        ("doc.csv", build_csv_bytes([{"name": "Ada"}]), "text/csv"),
        ("doc.json", build_json_bytes({"topic": "RAG"}), "application/json"),
    ],
)
def test_all_supported_formats_ingest(
    filename: str,
    content: bytes,
    content_type: str,
) -> None:
    vector_store = MagicMock()
    vector_store.find_by_payload.return_value = []
    embedding = MagicMock()
    embedding.provider_name = "huggingface"
    embedding.model_name = "test-model"
    embedding.embed_documents.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]

    from app.services.ingestion.service import DocumentIngestionService

    service = DocumentIngestionService(
        make_settings(embedding_dimension=2, chunk_size=100, chunk_overlap=0),
        vector_store,
        embedding,
    )
    result = service.ingest_document(
        filename=filename,
        content=content,
        content_type=content_type,
    )
    assert result.chunks_stored >= 1
    vector_store.add_vectors.assert_called_once()


def test_ingest_rejects_empty_file() -> None:
    from app.services.ingestion.service import DocumentIngestionService

    service = DocumentIngestionService(
        make_settings(embedding_dimension=2),
        MagicMock(),
        MagicMock(),
    )
    with pytest.raises(InvalidDocumentError) as exc_info:
        service.ingest_document(filename="empty.txt", content=b"", content_type="text/plain")
    assert exc_info.value.details.get("reason") == "empty_file"


def test_ingest_rejects_empty_pdf() -> None:
    from app.services.ingestion.service import DocumentIngestionService

    service = DocumentIngestionService(
        make_settings(embedding_dimension=2),
        MagicMock(),
        MagicMock(),
    )
    with pytest.raises(InvalidDocumentError):
        service.ingest_pdf(
            filename="empty.pdf",
            content=build_empty_pdf_bytes(),
            content_type="application/pdf",
        )


def test_rrf_deduplicates_deterministically() -> None:
    chunk_a = _chunk(chunk_id="doc-1:00000", score=0.9)
    chunk_b = _chunk(chunk_id="doc-1:00001", score=0.8)
    fused = reciprocal_rank_fusion(
        [[chunk_a, chunk_b], [chunk_a]],
        weights=[0.5, 0.5],
        limit=2,
    )
    assert [item.chunk_id for item in fused] == ["doc-1:00000", "doc-1:00001"]


def test_chunk_ids_are_deterministic_for_same_document() -> None:
    from app.services.chunking.service import ChunkingService
    from app.services.ingestion.base import ExtractedSection

    chunker = ChunkingService(make_settings(chunk_size=50, chunk_overlap=0), embedding_service=None)
    sections = [
        ExtractedSection(
            text="one two three four five six seven",
            section_index=0,
            page_number=1,
            section=None,
        )
    ]
    first = chunker.chunk_sections(
        sections,
        document_id="doc-fixed",
        filename="a.txt",
        file_type="txt",
        source="a.txt",
    )
    second = chunker.chunk_sections(
        sections,
        document_id="doc-fixed",
        filename="a.txt",
        file_type="txt",
        source="a.txt",
    )
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


# --- External failures ---


def test_hybrid_falls_back_when_keyword_search_fails() -> None:
    vector = MagicMock()
    vector.retrieve.return_value = [_chunk()]
    keyword = MagicMock()
    keyword.search.side_effect = OSError("index unavailable")

    hybrid = HybridRetrievalService(
        make_settings(hybrid_search_enabled=True),
        vector,
        keyword,
    )
    results = hybrid.retrieve("alpha beta")

    assert len(results) == 1
    assert results[0].chunk_id == "doc-1:00000"


def test_ingestion_continues_when_keyword_index_update_fails() -> None:
    from app.services.ingestion.service import DocumentIngestionService

    vector_store = MagicMock()
    vector_store.find_by_payload.return_value = []
    embedding = MagicMock()
    embedding.provider_name = "huggingface"
    embedding.model_name = "test-model"
    embedding.embed_documents.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    keyword = MagicMock()
    keyword.index_records.side_effect = OSError("disk full")

    service = DocumentIngestionService(
        make_settings(embedding_dimension=2, chunk_size=100, chunk_overlap=0),
        vector_store,
        embedding,
        keyword_search=keyword,
    )
    result = service.ingest_document(
        filename="notes.txt",
        content=build_txt_bytes("hello world"),
        content_type="text/plain",
    )
    assert result.chunks_stored >= 1
    vector_store.add_vectors.assert_called_once()


def test_multi_query_falls_back_when_generation_fails() -> None:
    inner = MagicMock()
    inner.retrieve.return_value = [_chunk()]
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("groq down")

    service = MultiQueryRetrievalService(
        make_settings(multi_query_enabled=True),
        inner,
        llm,
    )
    results = service.retrieve("question")

    assert len(results) == 1
    inner.retrieve.assert_called_once_with(
        "question",
        top_k=None,
        filters=None,
        score_threshold=None,
    )


# --- Citations ---


def test_citations_preserve_required_metadata_after_context_optimization() -> None:
    retrieval = MagicMock()
    retrieval.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00000", score=0.95),
        _chunk(
            chunk_id="doc-1:00000",
            text="duplicate",
            score=0.50,
            chunk_index=1,
        ),
        _chunk(
            chunk_id="doc-2:00000",
            text="unique zulu yankee xray",
            document_id="doc-2",
            filename="b.md",
            file_type="markdown",
            source="b.md",
            page_number=4,
            section="Methods",
            chunk_index=2,
            score=0.80,
        ),
    ]
    llm = MagicMock()
    llm.generate.return_value = "Answer"
    optimizer = ContextOptimizationService(
        make_settings(context_optimization_enabled=True, context_max_chunks=5)
    )
    rag = RAGService(
        retrieval_service=retrieval,
        llm_service=llm,
        context_optimizer=optimizer,
    )

    result = rag.answer("question")

    assert len(result.citations) == 2
    first, second = result.citations
    assert first.label == "S1"
    assert first.chunk_id == "doc-1:00000"
    assert first.document_id == "doc-1"
    assert first.filename == "a.pdf"
    assert first.page_number == 1
    assert first.section == "Intro"
    assert first.source == "a.pdf"

    assert second.label == "S2"
    assert second.chunk_id == "doc-2:00000"
    assert second.document_id == "doc-2"
    assert second.filename == "b.md"
    assert second.page_number == 4
    assert second.section == "Methods"


# --- Feature combinations ---


def test_advanced_features_enabled_together() -> None:
    settings = make_settings(
        query_transformation_enabled=True,
        multi_query_enabled=True,
        hybrid_search_enabled=True,
        context_optimization_enabled=True,
        context_max_chunks=5,
        hybrid_top_k=5,
        multi_query_count=2,
    )

    vector = MagicMock()
    vector.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00000", score=0.95),
        _chunk(chunk_id="doc-1:00001", text="other unique content here", score=0.85),
    ]
    keyword = MagicMock()
    keyword.search.return_value = [
        _chunk(chunk_id="doc-1:00001", text="other unique content here", score=6.0),
    ]
    hybrid = HybridRetrievalService(settings, vector, keyword)

    llm = MagicMock()
    llm.generate.side_effect = [
        "alpha beta\nalpha gamma",
        "Final answer",
    ]
    multi = MultiQueryRetrievalService(settings, hybrid, llm)

    transformer = MagicMock()
    transformer.transform.return_value = TransformedQuery(
        original_query="please explain alpha",
        transformed_query="alpha beta",
        was_transformed=True,
    )
    optimizer = ContextOptimizationService(settings)
    rag = RAGService(
        retrieval_service=multi,
        llm_service=llm,
        query_transformer=transformer,
        context_optimizer=optimizer,
    )

    result = rag.answer("please explain alpha")

    transformer.transform.assert_called_once()
    assert result.answer == "Final answer"
    assert result.citations


def test_cross_document_retrieval_with_filters() -> None:
    vector = MagicMock()
    vector.retrieve.return_value = [
        _chunk(document_id="doc-a", filename="a.pdf", chunk_id="doc-a:00000"),
        _chunk(document_id="doc-b", filename="b.pdf", chunk_id="doc-b:00000"),
    ]
    keyword = MagicMock()
    keyword.search.return_value = []

    hybrid = HybridRetrievalService(
        make_settings(hybrid_search_enabled=True),
        vector,
        keyword,
    )
    filters = RetrievalFilters.from_query(document_ids=["doc-a"])
    results = hybrid.retrieve("alpha", filters=filters)

    vector.retrieve.assert_called_once()
    assert vector.retrieve.call_args.kwargs["filters"] == filters
    assert len(results) >= 1


def test_context_optimizer_handles_no_results() -> None:
    optimizer = ContextOptimizationService(make_settings(context_optimization_enabled=True))
    result = optimizer.optimize([])
    assert result.chunks == []
    assert result.removed_count == 0


def test_context_optimizer_handles_low_relevance_with_min_score() -> None:
    optimizer = ContextOptimizationService(
        make_settings(context_optimization_enabled=True, context_min_score=0.8)
    )
    result = optimizer.optimize(
        [
            _chunk(chunk_id="doc-1:00000", score=0.95),
            _chunk(chunk_id="doc-1:00001", text="low", score=0.20),
        ]
    )
    assert len(result.chunks) == 1
    assert result.chunks[0].score == 0.95
