"""Detect uploaded document format from filename and content type."""

from __future__ import annotations

from pathlib import PurePath

from app.core.exceptions import InvalidDocumentError
from app.services.ingestion.base import (
    EXTENSION_TO_FILE_TYPE,
    FILE_TYPE_TO_CONTENT_TYPE,
    SUPPORTED_FILE_TYPES,
)


def detect_file_type(*, filename: str, content_type: str | None) -> tuple[str, str]:
    """
    Resolve ``(file_type, normalized_content_type)`` from upload metadata.

    Extension is the primary signal; content type is validated when explicit.
    """
    name = (filename or "").strip()
    if not name:
        raise InvalidDocumentError(
            "Filename is required",
            details={"reason": "missing_filename"},
        )

    suffix = PurePath(name).suffix.lower()
    if suffix not in EXTENSION_TO_FILE_TYPE:
        raise InvalidDocumentError(
            "Unsupported file type",
            details={
                "reason": "unsupported_file_type",
                "filename": name,
                "extension": suffix or None,
                "supported_extensions": sorted(EXTENSION_TO_FILE_TYPE),
            },
        )

    file_type = EXTENSION_TO_FILE_TYPE[suffix]
    if file_type not in SUPPORTED_FILE_TYPES:
        raise InvalidDocumentError(
            "Unsupported file type",
            details={"reason": "unsupported_file_type", "file_type": file_type},
        )

    expected_content_type = FILE_TYPE_TO_CONTENT_TYPE[file_type]
    if content_type is None or content_type == "" or content_type == "application/octet-stream":
        return file_type, expected_content_type

    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    allowed_types = _allowed_content_types_for(file_type)
    if media_type not in allowed_types:
        raise InvalidDocumentError(
            "Content type does not match file extension",
            details={
                "reason": "invalid_content_type",
                "content_type": media_type,
                "filename": name,
                "file_type": file_type,
                "allowed_content_types": sorted(allowed_types),
            },
        )
    return file_type, media_type


def _allowed_content_types_for(file_type: str) -> frozenset[str]:
    primary = FILE_TYPE_TO_CONTENT_TYPE[file_type]
    aliases: dict[str, frozenset[str]] = {
        "pdf": frozenset({primary, "application/x-pdf"}),
        "docx": frozenset(
            {
                primary,
                "application/msword",
                "application/vnd.ms-word",
            }
        ),
        "txt": frozenset({primary, "text/plain; charset=utf-8"}),
        "markdown": frozenset({primary, "text/plain", "text/x-markdown"}),
        "csv": frozenset({primary, "application/csv", "text/plain"}),
        "json": frozenset({primary, "text/json", "application/ld+json"}),
    }
    return aliases.get(file_type, frozenset({primary}))
