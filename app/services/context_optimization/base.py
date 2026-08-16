"""Context optimization abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.context_optimization.models import ContextOptimizationResult
from app.services.retrieval.service import RetrievedChunk


class ContextOptimizer(ABC):
    """Reduce retrieved context noise while preserving citations and metadata."""

    @abstractmethod
    def optimize(self, chunks: list[RetrievedChunk]) -> ContextOptimizationResult:
        """Select optimized chunks for prompt construction."""
