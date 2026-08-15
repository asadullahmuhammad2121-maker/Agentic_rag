"""Hugging Face embedding provider implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from huggingface_hub import InferenceClient

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.services.embeddings.base import EmbeddingService
from app.utils.checksum import sha256_digest

logger = get_logger(__name__)


class HuggingFaceEmbeddingService(EmbeddingService):
    """Hugging Face Inference API embedding service with batched requests."""

    def __init__(self, settings: Settings, client: InferenceClient | None = None) -> None:
        self._settings = settings
        self._model = settings.huggingface_embedding_model
        self._dimension = settings.embedding_dimension
        self._batch_size = settings.embedding_batch_size
        # Request-scoped / process-scoped cache to avoid regenerating identical embeddings.
        self._cache: dict[str, list[float]] = {}
        try:
            self._client = client or InferenceClient(
                model=self._model,
                token=settings.huggingface_api_key.get_secret_value(),
                timeout=settings.huggingface_timeout_seconds,
            )
        except Exception as exc:
            logger.error(
                "huggingface_client_init_failed",
                extra={
                    "operation": "init",
                    "provider": "huggingface",
                    "error_type": type(exc).__name__,
                },
            )
            raise ProviderError(
                "Failed to initialize Hugging Face client",
                provider="huggingface",
            ) from exc

        logger.info(
            "huggingface_embedding_service_initialized",
            extra={
                "operation": "init",
                "provider": "huggingface",
                "model": self._model,
                "dimension": self._dimension,
                "batch_size": self._batch_size,
            },
        )

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def health_check(self) -> bool:
        """Confirm the client and model configuration are present."""
        api_key = self._settings.huggingface_api_key.get_secret_value()
        healthy = bool(api_key) and bool(self._model) and self._client is not None
        logger.debug(
            "huggingface_health_check",
            extra={
                "operation": "health_check",
                "provider": "huggingface",
                "healthy": healthy,
            },
        )
        return healthy

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts using configurable batching and an in-memory cache."""
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        pending_keys: list[str] = []
        pending_texts: list[str] = []
        key_to_indexes: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            normalized = text.strip()
            if not normalized:
                raise ProviderError(
                    "Cannot embed empty text",
                    provider="huggingface",
                    details={"reason": "empty_text"},
                )
            cache_key = sha256_digest(normalized.encode("utf-8"))
            cached = self._cache.get(cache_key)
            if cached is not None:
                vectors[index] = cached
                continue

            if cache_key in key_to_indexes:
                key_to_indexes[cache_key].append(index)
                continue

            key_to_indexes[cache_key] = [index]
            pending_keys.append(cache_key)
            pending_texts.append(normalized)

        cache_hits = sum(1 for vector in vectors if vector is not None)
        if pending_texts:
            embedded = self._embed_batched(pending_texts)
            for cache_key, vector in zip(pending_keys, embedded, strict=True):
                self._cache[cache_key] = vector
                for index in key_to_indexes[cache_key]:
                    vectors[index] = vector

        logger.info(
            "documents_embedded",
            extra={
                "operation": "embed_documents",
                "provider": "huggingface",
                "model": self._model,
                "text_count": len(texts),
                "cache_hits": cache_hits,
                "unique_api_texts": len(pending_texts),
                "batch_size": self._batch_size,
            },
        )
        result: list[list[float]] = []
        for maybe_vector in vectors:
            if maybe_vector is None:
                raise ProviderError(
                    "Internal embedding cache inconsistency",
                    provider="huggingface",
                    details={"reason": "cache_inconsistency"},
                )
            result.append(maybe_vector)
        return result

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (ready for Phase 1D retrieval)."""
        vectors = self.embed_documents([text])
        return vectors[0]

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_number, start in enumerate(range(0, len(texts), self._batch_size), start=1):
            batch = texts[start : start + self._batch_size]
            logger.debug(
                "embedding_batch_started",
                extra={
                    "operation": "embed_documents",
                    "provider": "huggingface",
                    "batch_number": batch_number,
                    "total_batches": total_batches,
                    "batch_count": len(batch),
                },
            )
            for text in batch:
                results.append(self._embed_one(text))

        return results

    def _embed_one(self, text: str) -> list[float]:
        try:
            raw = self._client.feature_extraction(text, normalize=True)
        except Exception as exc:
            logger.error(
                "huggingface_embedding_failed",
                extra={
                    "operation": "embed_one",
                    "provider": "huggingface",
                    "model": self._model,
                    "error_type": type(exc).__name__,
                    "text_length": len(text),
                },
            )
            raise ProviderError(
                "Hugging Face embedding request failed",
                provider="huggingface",
                details={"reason": "api_failure", "error_type": type(exc).__name__},
            ) from exc

        try:
            return self._normalize_vector(raw)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                "Failed to parse embedding response",
                provider="huggingface",
                details={"reason": "invalid_response"},
            ) from exc

    def _normalize_vector(self, raw: Any) -> list[float]:
        # huggingface_hub returns numpy arrays; avoid importing numpy at module import time
        # if unavailable by using duck-typing.
        values: list[Any]
        if hasattr(raw, "tolist"):
            values = raw.tolist()
        elif isinstance(raw, list):
            values = raw
        else:
            values = list(raw)

        # Some models return token-level vectors [[d], ...]; mean-pool to one vector.
        if values and isinstance(values[0], (list, tuple)):
            token_vectors = [list(map(float, row)) for row in values]
            dim = len(token_vectors[0])
            pooled = [0.0] * dim
            for row in token_vectors:
                if len(row) != dim:
                    raise ProviderError(
                        "Inconsistent embedding token dimensions",
                        provider="huggingface",
                        details={"reason": "dimension_mismatch"},
                    )
                for i, value in enumerate(row):
                    pooled[i] += value
            scale = 1.0 / len(token_vectors)
            vector = [value * scale for value in pooled]
        else:
            vector = [float(value) for value in values]

        if len(vector) != self._dimension:
            raise ProviderError(
                "Embedding dimension mismatch",
                provider="huggingface",
                details={
                    "reason": "dimension_mismatch",
                    "expected": self._dimension,
                    "actual": len(vector),
                },
            )
        return vector
