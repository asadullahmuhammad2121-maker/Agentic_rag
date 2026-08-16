"""Unit tests for retrieval filter models."""

from __future__ import annotations

import pytest

from app.core.exceptions import QueryError
from app.services.retrieval.filters import RetrievalFilters
from app.vector_store.filters import PayloadFilter


def test_empty_filters_return_none() -> None:
    assert RetrievalFilters.from_query() is None


def test_single_document_id_uses_exact_match() -> None:
    filters = RetrievalFilters.from_query(document_ids=["doc-1"])
    assert filters is not None
    payload_filter = filters.to_payload_filter()
    assert payload_filter == PayloadFilter(exact={"document_id": "doc-1"})


def test_multiple_document_ids_use_any_match() -> None:
    filters = RetrievalFilters.from_query(document_ids=["doc-1", "doc-2"])
    assert filters is not None
    payload_filter = filters.to_payload_filter()
    assert payload_filter == PayloadFilter(any_of={"document_id": ("doc-1", "doc-2")})


def test_combined_filters_and_across_fields() -> None:
    filters = RetrievalFilters.from_query(
        document_ids=["doc-1"],
        file_types=["pdf", "txt"],
        sections=["Introduction"],
    )
    assert filters is not None
    payload_filter = filters.to_payload_filter()
    assert payload_filter == PayloadFilter(
        exact={"document_id": "doc-1", "section": "Introduction"},
        any_of={"file_type": ("pdf", "txt")},
    )


def test_filename_filter_supported() -> None:
    filters = RetrievalFilters.from_query(filenames=["report.pdf"])
    assert filters is not None
    payload_filter = filters.to_payload_filter()
    assert payload_filter == PayloadFilter(exact={"filename": "report.pdf"})


def test_legacy_filters_merge_into_structured_filters() -> None:
    filters = RetrievalFilters.from_query(
        file_types=["pdf"],
        legacy_filters={"document_id": "doc-legacy"},
    )
    assert filters is not None
    payload_filter = filters.to_payload_filter()
    assert payload_filter == PayloadFilter(
        exact={"document_id": "doc-legacy", "file_type": "pdf"},
    )


def test_rejects_empty_filter_values() -> None:
    with pytest.raises(QueryError) as exc_info:
        RetrievalFilters.from_query(document_ids=["   "])
    assert exc_info.value.details.get("reason") == "invalid_filter"


def test_matches_payload_respects_all_fields() -> None:
    filters = RetrievalFilters(
        document_ids=("doc-1",),
        filenames=("a.pdf",),
        file_types=("pdf",),
        sections=("Intro",),
    )
    matching = {
        "document_id": "doc-1",
        "filename": "a.pdf",
        "file_type": "pdf",
        "section": "Intro",
    }
    non_matching = {
        "document_id": "doc-2",
        "filename": "a.pdf",
        "file_type": "pdf",
        "section": "Intro",
    }

    assert filters.matches_payload(matching) is True
    assert filters.matches_payload(non_matching) is False
