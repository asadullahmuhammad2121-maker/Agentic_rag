"""Checksum helpers."""

from __future__ import annotations

import hashlib


def sha256_digest(data: bytes) -> str:
    """Return a hex SHA-256 digest for ``data``."""
    return hashlib.sha256(data).hexdigest()
