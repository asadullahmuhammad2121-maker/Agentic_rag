"""Embedding service package."""

from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.huggingface import HuggingFaceEmbeddingService

__all__ = ["EmbeddingService", "HuggingFaceEmbeddingService"]
