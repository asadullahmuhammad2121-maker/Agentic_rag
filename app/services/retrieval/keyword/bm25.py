"""BM25 keyword search over ingested document chunks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.services.retrieval.chunk_mapping import payload_to_retrieved_chunk
from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.keyword.base import KeywordSearch
from app.services.retrieval.service import RetrievedChunk
from app.vector_store.base import VectorRecord

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")
_INDEX_VERSION = 1


@dataclass(slots=True, frozen=True)
class IndexedChunk:
    """A chunk stored in the keyword index."""

    chunk_id: str
    text: str
    payload: dict[str, Any]


class BM25Scorer:
    """Okapi BM25 scorer for a tokenized corpus."""

    def __init__(
        self,
        corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._corpus = corpus
        self._doc_len = [len(doc) for doc in corpus]
        self._avgdl = sum(self._doc_len) / len(corpus) if corpus else 0.0
        self._df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(corpus)

    def score(self, query_tokens: list[str]) -> list[float]:
        """Return BM25 scores for each document in the corpus."""
        if not self._corpus or not query_tokens:
            return [0.0] * self._n

        scores = [0.0] * self._n
        for term in query_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for index, doc in enumerate(self._corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                doc_len = self._doc_len[index]
                denom = tf + self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
                scores[index] += idf * (tf * (self._k1 + 1.0)) / denom
        return scores


class BM25KeywordSearch(KeywordSearch):
    """Persistent in-memory BM25 index backed by a JSON file."""

    def __init__(self, index_path: str | Path) -> None:
        self._index_path = Path(index_path)
        self._chunks: list[IndexedChunk] = []
        self._chunk_index_by_id: dict[str, int] = {}
        self._scorer: BM25Scorer | None = None
        self._load()

    def index_records(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        updated = 0
        for record in records:
            text = str(record.payload.get("text", "")).strip()
            if not text:
                continue
            chunk_id = str(record.payload.get("chunk_id") or record.id)
            indexed = IndexedChunk(chunk_id=chunk_id, text=text, payload=dict(record.payload))
            existing_index = self._chunk_index_by_id.get(chunk_id)
            if existing_index is not None:
                self._chunks[existing_index] = indexed
            else:
                self._chunk_index_by_id[chunk_id] = len(self._chunks)
                self._chunks.append(indexed)
            updated += 1

        self._rebuild_scorer()
        self._save()
        logger.info(
            "keyword_index_updated",
            extra={
                "operation": "index_records",
                "records_received": len(records),
                "records_indexed": updated,
                "total_chunks": len(self._chunks),
            },
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        normalized = query.strip()
        if not normalized or top_k <= 0 or not self._chunks or self._scorer is None:
            return []

        query_tokens = tokenize(normalized)
        if not query_tokens:
            return []

        raw_scores = self._scorer.score(query_tokens)
        ranked_indices = sorted(
            range(len(raw_scores)),
            key=lambda index: raw_scores[index],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = raw_scores[index]
            if score <= 0.0:
                break
            chunk = self._chunks[index]
            if filters is not None and not filters.matches_payload(chunk.payload):
                continue
            results.append(payload_to_retrieved_chunk(chunk.chunk_id, score, chunk.payload))
            if len(results) >= top_k:
                break
        return results

    def _rebuild_scorer(self) -> None:
        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        self._scorer = BM25Scorer(tokenized) if tokenized else None

    def _load(self) -> None:
        if not self._index_path.exists():
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            return

        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "keyword_index_load_failed",
                extra={
                    "operation": "load_keyword_index",
                    "path": str(self._index_path),
                    "error_type": type(exc).__name__,
                },
            )
            return

        if raw.get("version") != _INDEX_VERSION:
            logger.warning(
                "keyword_index_version_mismatch",
                extra={
                    "operation": "load_keyword_index",
                    "path": str(self._index_path),
                    "expected_version": _INDEX_VERSION,
                    "actual_version": raw.get("version"),
                },
            )
            return

        loaded: list[IndexedChunk] = []
        for item in raw.get("chunks", []):
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id", "")).strip()
            text = str(item.get("text", "")).strip()
            payload = item.get("payload")
            if not chunk_id or not text or not isinstance(payload, dict):
                continue
            loaded.append(IndexedChunk(chunk_id=chunk_id, text=text, payload=payload))

        self._chunks = loaded
        self._chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(loaded)}
        self._rebuild_scorer()

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "payload": chunk.payload,
                }
                for chunk in self._chunks
            ],
        }
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring."""
    return _TOKEN_PATTERN.findall(text.casefold())
