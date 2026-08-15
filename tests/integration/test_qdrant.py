"""Integration tests for Qdrant connectivity (optional live service)."""

from __future__ import annotations

import os

import pytest

from app.vector_store.qdrant import QdrantVectorStore
from tests.conftest import make_settings

pytestmark = pytest.mark.integration


def _qdrant_available() -> bool:
    """Return True when a live Qdrant instance responds on QDRANT_URL."""
    try:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url, timeout=2, check_compatibility=False)
        client.get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant is not reachable (start via docker compose)",
)


@requires_qdrant
def test_qdrant_health_check_live() -> None:
    store = QdrantVectorStore(
        make_settings(qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333")),
    )
    assert store.health_check() is True


@requires_qdrant
def test_qdrant_collection_lifecycle_live() -> None:
    collection = "phase1a_health_probe"
    settings = make_settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection_name=collection,
        embedding_dimension=4,
    )
    store = QdrantVectorStore(settings)
    store.create_collection(collection, vector_size=4)
    store.delete_collection(collection)
