"""Identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid4())


def new_point_id() -> str:
    """Generate a unique vector-store point identifier."""
    return str(uuid4())
