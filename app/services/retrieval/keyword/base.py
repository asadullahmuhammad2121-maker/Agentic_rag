"""Keyword search abstraction for hybrid retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.retrieval.filters import RetrievalFilters
from app.services.retrieval.service import RetrievedChunk
from app.vector_store.base import VectorRecord


class KeywordSearch(ABC):
    """Search indexed document chunks using keyword/BM25 scoring."""

    @abstractmethod
    def index_records(self, records: list[VectorRecord]) -> None:
        """Add or update chunks in the keyword index."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return top keyword matches for ``query``."""
