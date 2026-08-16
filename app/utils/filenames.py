"""Safe filename handling for uploaded documents."""

from __future__ import annotations

from pathlib import PurePath

from app.core.exceptions import InvalidDocumentError


def sanitize_upload_filename(filename: str) -> str:
    """
    Normalize an uploaded filename to its basename.

    Strips directory components to prevent path traversal and rejects unsafe names.
    """
    name = (filename or "").strip()
    if not name:
        raise InvalidDocumentError(
            "Filename is required",
            details={"reason": "missing_filename"},
        )

    basename = PurePath(name).name
    if not basename or basename in {".", ".."}:
        raise InvalidDocumentError(
            "Invalid filename",
            details={"reason": "invalid_filename", "filename": name},
        )
    if "\x00" in basename:
        raise InvalidDocumentError(
            "Invalid filename",
            details={"reason": "invalid_filename", "filename": name},
        )
    return basename
