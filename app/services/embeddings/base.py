"""Provider-independent embedding service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingService(ABC):
    """Abstract embedding provider used by future ingestion / retrieval phases."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a stable provider identifier (e.g. 'huggingface')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured embedding model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the expected embedding dimensionality."""

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the provider client is configured for later embedding calls.

        Phase 1A: validate client construction without calling the API.
        """

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a batch of documents.

        Implemented by providers in Phase 1C.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Implemented by providers in Phase 1C (used by retrieval in Phase 1D).
        """
