"""Identifier helpers."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


def new_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid4())


def new_point_id() -> str:
    """Generate a unique vector-store point identifier."""
    return str(uuid4())


def normalize_point_id(point_id: str | UUID) -> str | int:
    """Return a Qdrant-compatible point ID for a logical chunk identifier."""
    if isinstance(point_id, UUID):
        return str(point_id)
    if isinstance(point_id, str):
        if point_id.isdigit():
            return int(point_id)
        try:
            UUID(point_id)
        except ValueError:
            return str(uuid5(NAMESPACE_URL, point_id))
        return point_id
    return point_id
