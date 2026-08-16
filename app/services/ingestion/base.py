"""Normalized document models and parser abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

SUPPORTED_FILE_TYPES: Final[frozenset[str]] = frozenset(
    {"pdf", "docx", "txt", "markdown", "csv", "json"}
)

EXTENSION_TO_FILE_TYPE: Final[dict[str, str]] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
    ".csv": "csv",
    ".json": "json",
}

FILE_TYPE_TO_CONTENT_TYPE: Final[dict[str, str]] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "markdown": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
}


@dataclass(slots=True, frozen=True)
class ExtractedSection:
    """A normalized section of extracted document content."""

    text: str
    section_index: int
    page_number: int | None = None
    section: str | None = None


@dataclass(slots=True, frozen=True)
class NormalizedDocument:
    """Format-agnostic extraction result ready for chunking."""

    sections: list[ExtractedSection]
    file_type: str
    source: str
    page_count: int
    section_count: int


@dataclass(slots=True, frozen=True)
class ExtractedPage:
    """Backward-compatible PDF page model."""

    page_number: int
    text: str


@dataclass(slots=True, frozen=True)
class ExtractedDocument:
    """Backward-compatible PDF extraction result."""

    pages: list[ExtractedPage]
    page_count: int


class DocumentParser(ABC):
    """Parse uploaded bytes into a normalized document representation."""

    @property
    @abstractmethod
    def file_type(self) -> str:
        """Short format identifier (e.g. pdf, docx, txt)."""

    @abstractmethod
    def parse(self, content: bytes, *, filename: str) -> NormalizedDocument:
        """Extract text and metadata from ``content``."""
