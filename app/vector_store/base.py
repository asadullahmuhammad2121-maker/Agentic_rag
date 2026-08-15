"""Provider-independent vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(slots=True)
class VectorRecord:
    """A single vector payload for upsert operations."""

    id: str | UUID
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """A single nearest-neighbor search hit."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract vector database used by future ingestion / retrieval phases."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the vector store is reachable."""

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        *,
        distance: str = "Cosine",
    ) -> None:
        """Create a collection if it does not already exist."""

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by name."""

    @abstractmethod
    def add_vectors(
        self,
        collection_name: str,
        records: list[VectorRecord],
    ) -> None:
        """Upsert vectors into a collection."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Run a similarity search against a collection.

        ``filters`` is an optional mapping of payload field → exact match value.
        """

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: list[str | UUID],
    ) -> None:
        """Delete vectors by identifier."""

    @abstractmethod
    def find_by_payload(
        self,
        collection_name: str,
        conditions: dict[str, Any],
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Find points whose payload matches exact key/value conditions.

        Used for duplicate detection and metadata lookups without vector search.
        """

    @abstractmethod
    def ensure_payload_index(
        self,
        collection_name: str,
        field_name: str,
        *,
        field_schema: str = "keyword",
    ) -> None:
        """Ensure a payload field is indexed for efficient filtering."""
