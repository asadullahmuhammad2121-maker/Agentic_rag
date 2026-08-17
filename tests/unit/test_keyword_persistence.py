"""Unit tests for BM25 keyword index persistence and multi-replica safety."""

from __future__ import annotations

from pathlib import Path

from app.services.retrieval.keyword.bm25 import BM25KeywordSearch
from app.vector_store.base import VectorRecord


def _record(*, chunk_id: str, text: str) -> VectorRecord:
    return VectorRecord(
        id=chunk_id,
        vector=[0.1, 0.2],
        payload={
            "chunk_id": chunk_id,
            "text": text,
            "document_id": "doc-1",
            "filename": "a.pdf",
            "file_type": "pdf",
            "source": "a.pdf",
            "page_number": 1,
            "section": None,
            "chunk_index": 0,
            "chunking_strategy": "fixed",
        },
    )


def test_keyword_index_persists_across_instances(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    writer = BM25KeywordSearch(index_path)
    writer.index_records([_record(chunk_id="c1", text="persistent keyword index")])

    reader = BM25KeywordSearch(index_path)
    results = reader.search("persistent keyword", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_second_replica_sees_updates_from_first_replica(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    replica_a = BM25KeywordSearch(index_path)
    replica_b = BM25KeywordSearch(index_path)

    replica_a.index_records([_record(chunk_id="c1", text="shared hybrid retrieval")])
    results = replica_b.search("shared hybrid", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_keyword_index_health_check(tmp_path: Path) -> None:
    index_path = tmp_path / "nested" / "index.json"
    search = BM25KeywordSearch(index_path)
    assert search.health_check() is True
