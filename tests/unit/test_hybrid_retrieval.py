"""Unit tests for hybrid retrieval (Phase 2F)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.hybrid import HybridRetrievalService
from app.services.retrieval.keyword.bm25 import BM25KeywordSearch
from app.services.retrieval.multi_query import MultiQueryRetrievalService
from app.services.retrieval.service import RetrievalService, RetrievedChunk
from app.vector_store.base import VectorRecord
from tests.conftest import make_settings


def _chunk(
    *,
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    filename: str = "a.pdf",
    file_type: str = "pdf",
    score: float = 1.0,
    page_number: int = 1,
    section: str | None = None,
    chunk_index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        source=filename,
        page_number=page_number,
        section=section,
        chunk_index=chunk_index,
        chunking_strategy="fixed",
        score=score,
    )


def _record(
    *,
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    filename: str = "a.pdf",
    file_type: str = "pdf",
    page_number: int = 1,
    section: str | None = None,
    chunk_index: int = 0,
) -> VectorRecord:
    return VectorRecord(
        id=chunk_id,
        vector=[0.1, 0.2],
        payload={
            "text": text,
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "source": filename,
            "page_number": page_number,
            "section": section,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "chunking_strategy": "fixed",
        },
    )


@pytest.fixture
def keyword_index(tmp_path: Path) -> BM25KeywordSearch:
    return BM25KeywordSearch(tmp_path / "keyword_index.json")


@pytest.fixture
def vector_retrieval() -> MagicMock:
    return MagicMock(spec=RetrievalService)


@pytest.fixture
def hybrid_disabled(
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> HybridRetrievalService:
    settings = make_settings(hybrid_search_enabled=False, retrieval_top_k=5)
    return HybridRetrievalService(settings, vector_retrieval, keyword_index)


@pytest.fixture
def hybrid_enabled(
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> HybridRetrievalService:
    settings = make_settings(
        hybrid_search_enabled=True,
        hybrid_top_k=10,
        vector_search_weight=0.5,
        keyword_search_weight=0.5,
        retrieval_top_k=5,
    )
    return HybridRetrievalService(settings, vector_retrieval, keyword_index)


def test_vector_only_mode_delegates_to_vector_service(
    hybrid_disabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    expected = [_chunk(chunk_id="doc-1:00001", text="vector hit", score=0.9)]
    vector_retrieval.retrieve.return_value = expected

    result = hybrid_disabled.retrieve("exact keyword", top_k=3)

    assert result == expected
    vector_retrieval.retrieve.assert_called_once_with(
        "exact keyword",
        top_k=3,
        filters=None,
        score_threshold=None,
    )
    assert keyword_index.search("exact keyword", top_k=3) == []


def test_keyword_search_ranks_exact_terms(keyword_index: BM25KeywordSearch) -> None:
    keyword_index.index_records(
        [
            _record(chunk_id="doc-1:00001", text="machine learning basics"),
            _record(chunk_id="doc-1:00002", text="database indexing strategies"),
            _record(chunk_id="doc-2:00001", text="machine learning advanced topics"),
        ]
    )

    results = keyword_index.search("machine learning", top_k=2)

    assert len(results) == 2
    assert all("machine" in result.text for result in results)
    assert results[0].chunk_id in {"doc-1:00001", "doc-2:00001"}


def test_keyword_search_applies_document_filters(keyword_index: BM25KeywordSearch) -> None:
    keyword_index.index_records(
        [
            _record(
                chunk_id="doc-1:00001",
                text="alpha beta gamma",
                document_id="doc-1",
            ),
            _record(
                chunk_id="doc-2:00001",
                text="alpha beta gamma",
                document_id="doc-2",
            ),
        ]
    )
    filters = RetrievalFilters.from_query(document_ids=["doc-2"])

    results = keyword_index.search("alpha beta", top_k=5, filters=filters)

    assert len(results) == 1
    assert results[0].document_id == "doc-2"


def test_keyword_search_preserves_metadata(keyword_index: BM25KeywordSearch) -> None:
    keyword_index.index_records(
        [
            _record(
                chunk_id="doc-1:00003",
                text="retrieval augmented generation",
                filename="notes.md",
                file_type="markdown",
                page_number=4,
                section="Intro",
                chunk_index=3,
            )
        ]
    )

    chunk = keyword_index.search("retrieval augmented", top_k=1)[0]

    assert chunk.chunk_id == "doc-1:00003"
    assert chunk.filename == "notes.md"
    assert chunk.file_type == "markdown"
    assert chunk.page_number == 4
    assert chunk.section == "Intro"
    assert chunk.chunk_index == 3
    assert chunk.source == "notes.md"


def test_keyword_search_empty_results(keyword_index: BM25KeywordSearch) -> None:
    keyword_index.index_records([_record(chunk_id="doc-1:00001", text="hello world")])

    assert keyword_index.search("quantum physics", top_k=5) == []
    assert keyword_index.search("   ", top_k=5) == []


def test_rrf_fuses_and_deduplicates_by_chunk_id() -> None:
    shared = _chunk(chunk_id="doc-1:00001", text="shared", score=0.8)
    vector_only = _chunk(chunk_id="doc-1:00002", text="vector only", score=0.7)
    keyword_only = _chunk(chunk_id="doc-1:00003", text="keyword only", score=6.0)

    fused = reciprocal_rank_fusion(
        [
            [shared, vector_only],
            [shared, keyword_only],
        ],
        weights=[0.5, 0.5],
        limit=3,
    )

    chunk_ids = [chunk.chunk_id for chunk in fused]
    assert chunk_ids[0] == "doc-1:00001"
    assert set(chunk_ids) == {"doc-1:00001", "doc-1:00002", "doc-1:00003"}
    assert fused[0].score >= fused[1].score >= fused[2].score


def test_rrf_respects_weights() -> None:
    vector_chunk = _chunk(chunk_id="doc-1:00001", text="vector", score=0.9)
    keyword_chunk = _chunk(chunk_id="doc-1:00002", text="keyword", score=5.0)

    vector_heavy = reciprocal_rank_fusion(
        [[vector_chunk], [keyword_chunk]],
        weights=[1.0, 0.1],
        limit=2,
    )
    keyword_heavy = reciprocal_rank_fusion(
        [[vector_chunk], [keyword_chunk]],
        weights=[0.1, 1.0],
        limit=2,
    )

    assert vector_heavy[0].chunk_id == "doc-1:00001"
    assert keyword_heavy[0].chunk_id == "doc-1:00002"


def test_hybrid_retrieval_combines_vector_and_keyword(
    hybrid_enabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    vector_retrieval.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00001", text="vector chunk", score=0.95)
    ]
    keyword_index.index_records(
        [
            _record(chunk_id="doc-1:00002", text="exact keyword match"),
            _record(chunk_id="doc-1:00001", text="vector chunk overlap"),
        ]
    )

    results = hybrid_enabled.retrieve("exact keyword")

    assert len(results) == 2
    assert {chunk.chunk_id for chunk in results} == {"doc-1:00001", "doc-1:00002"}
    vector_retrieval.retrieve.assert_called_once()


def test_hybrid_retrieval_uses_hybrid_top_k_default(
    hybrid_enabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
) -> None:
    vector_retrieval.retrieve.return_value = []

    hybrid_enabled.retrieve("query")

    assert vector_retrieval.retrieve.call_args.kwargs["top_k"] == 10


def test_hybrid_retrieval_handles_empty_vector_and_keyword(
    hybrid_enabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
) -> None:
    vector_retrieval.retrieve.return_value = []

    assert hybrid_enabled.retrieve("missing topic") == []


def test_hybrid_retrieval_keyword_only_when_vector_empty(
    hybrid_enabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    vector_retrieval.retrieve.return_value = []
    keyword_index.index_records(
        [_record(chunk_id="doc-1:00001", text="keyword only retrieval")]
    )

    results = hybrid_enabled.retrieve("keyword only", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "doc-1:00001"


def test_hybrid_retrieval_passes_filters(
    hybrid_enabled: HybridRetrievalService,
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    vector_retrieval.retrieve.return_value = []
    keyword_index.index_records(
        [
            _record(chunk_id="doc-1:00001", text="filtered alpha", document_id="doc-1"),
            _record(chunk_id="doc-2:00001", text="filtered alpha", document_id="doc-2"),
        ]
    )
    filters = RetrievalFilters.from_query(filenames=["b.txt"])

    hybrid_enabled.retrieve("filtered alpha", filters=filters)

    vector_retrieval.retrieve.assert_called_once()
    assert vector_retrieval.retrieve.call_args.kwargs["filters"] == filters


def test_multi_document_keyword_retrieval(keyword_index: BM25KeywordSearch) -> None:
    keyword_index.index_records(
        [
            _record(
                chunk_id="doc-a:00001",
                text="project atlas launch timeline",
                document_id="doc-a",
                filename="a.pdf",
            ),
            _record(
                chunk_id="doc-b:00001",
                text="project atlas budget review",
                document_id="doc-b",
                filename="b.pdf",
            ),
        ]
    )

    results = keyword_index.search("project atlas", top_k=5)

    assert len(results) == 2
    assert {chunk.document_id for chunk in results} == {"doc-a", "doc-b"}


def test_ingestion_indexes_keyword_store(keyword_index: BM25KeywordSearch) -> None:
    records = [
        _record(chunk_id="doc-1:00001", text="indexed during ingestion"),
        _record(chunk_id="doc-1:00002", text="another indexed chunk"),
    ]

    keyword_index.index_records(records)

    results = keyword_index.search("indexed ingestion", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "doc-1:00001"


def test_multi_query_uses_hybrid_backend_when_enabled(
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    settings = make_settings(
        hybrid_search_enabled=True,
        multi_query_enabled=False,
        hybrid_top_k=5,
    )
    hybrid = HybridRetrievalService(settings, vector_retrieval, keyword_index)
    llm = MagicMock()
    service = MultiQueryRetrievalService(settings, hybrid, llm)
    vector_retrieval.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00001", text="hybrid path", score=0.8)
    ]

    results = service.retrieve("hybrid path")

    assert len(results) == 1
    vector_retrieval.retrieve.assert_called_once()


def test_multi_query_with_hybrid_and_generation_enabled(
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    settings = make_settings(
        hybrid_search_enabled=True,
        multi_query_enabled=True,
        multi_query_count=3,
        hybrid_top_k=5,
    )
    hybrid = HybridRetrievalService(settings, vector_retrieval, keyword_index)
    llm = MagicMock()
    llm.generate.return_value = "variant one\nvariant two\nvariant three"
    vector_retrieval.retrieve.side_effect = [
        [_chunk(chunk_id="doc-1:00001", text="one", score=0.9)],
        [_chunk(chunk_id="doc-1:00002", text="two", score=0.8)],
        [_chunk(chunk_id="doc-1:00001", text="one duplicate", score=0.7)],
    ]
    keyword_index.index_records(
        [
            _record(chunk_id="doc-1:00001", text="one duplicate"),
            _record(chunk_id="doc-1:00002", text="two"),
        ]
    )

    service = MultiQueryRetrievalService(settings, hybrid, llm)
    results = service.retrieve("original question")

    assert vector_retrieval.retrieve.call_count == 3
    assert len(results) <= 5
    assert len({chunk.chunk_id for chunk in results}) == len(results)


def test_query_transformation_compatible_with_hybrid(
    vector_retrieval: MagicMock,
    keyword_index: BM25KeywordSearch,
) -> None:
    settings = make_settings(hybrid_search_enabled=True, hybrid_top_k=3)
    hybrid = HybridRetrievalService(settings, vector_retrieval, keyword_index)
    vector_retrieval.retrieve.return_value = [
        _chunk(chunk_id="doc-1:00001", text="transformed retrieval", score=0.88)
    ]
    keyword_index.index_records(
        [_record(chunk_id="doc-1:00002", text="transformed retrieval keyword")]
    )

    results = hybrid.retrieve("rewritten query for retrieval")

    assert len(results) == 2
    vector_retrieval.retrieve.assert_called_once_with(
        "rewritten query for retrieval",
        top_k=3,
        filters=None,
        score_threshold=None,
    )
