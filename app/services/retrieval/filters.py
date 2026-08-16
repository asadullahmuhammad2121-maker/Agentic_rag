"""Retrieval filter models and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import QueryError
from app.vector_store.filters import PayloadFilter


def _clean_filter_values(values: list[str] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    cleaned = tuple(dict.fromkeys(item.strip() for item in values if item and item.strip()))
    if values and not cleaned:
        raise QueryError(
            f"{field_name} must not contain empty values",
            details={"reason": "invalid_filter", "field": field_name},
        )
    return cleaned


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Validated metadata filters for document-aware retrieval."""

    document_ids: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    file_types: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()

    @classmethod
    def from_query(
        cls,
        *,
        document_ids: list[str] | None = None,
        filenames: list[str] | None = None,
        file_types: list[str] | None = None,
        sections: list[str] | None = None,
        legacy_filters: dict[str, str | int] | None = None,
    ) -> RetrievalFilters | None:
        """Build validated filters from query parameters."""
        filters = cls(
            document_ids=_clean_filter_values(document_ids, field_name="document_ids"),
            filenames=_clean_filter_values(filenames, field_name="filenames"),
            file_types=_clean_filter_values(file_types, field_name="file_types"),
            sections=_clean_filter_values(sections, field_name="sections"),
        )
        if legacy_filters:
            filters = filters.merge_legacy(legacy_filters)
        return None if filters.is_empty() else filters

    def merge_legacy(self, legacy_filters: dict[str, str | int]) -> RetrievalFilters:
        """Merge backward-compatible exact-match filters."""
        if not legacy_filters:
            return self

        document_ids = self.document_ids
        filenames = self.filenames
        file_types = self.file_types
        sections = self.sections
        exact = PayloadFilter.from_legacy_dict(legacy_filters)

        if "document_id" in exact.exact:
            document_ids = document_ids + (str(exact.exact["document_id"]),)
        if "document_id" in exact.any_of:
            document_ids = document_ids + exact.any_of["document_id"]
        if "filename" in exact.exact:
            filenames = filenames + (str(exact.exact["filename"]),)
        if "filename" in exact.any_of:
            filenames = filenames + exact.any_of["filename"]
        if "file_type" in exact.exact:
            file_types = file_types + (str(exact.exact["file_type"]),)
        if "file_type" in exact.any_of:
            file_types = file_types + exact.any_of["file_type"]
        if "section" in exact.exact:
            sections = sections + (str(exact.exact["section"]),)
        if "section" in exact.any_of:
            sections = sections + exact.any_of["section"]

        return RetrievalFilters(
            document_ids=_dedupe(document_ids),
            filenames=_dedupe(filenames),
            file_types=_dedupe(file_types),
            sections=_dedupe(sections),
        )

    def is_empty(self) -> bool:
        return not (
            self.document_ids or self.filenames or self.file_types or self.sections
        )

    def to_payload_filter(self) -> PayloadFilter | None:
        if self.is_empty():
            return None

        exact: dict[str, str | int] = {}
        any_of: dict[str, tuple[str, ...]] = {}

        _assign_filter("document_id", self.document_ids, exact=exact, any_of=any_of)
        _assign_filter("filename", self.filenames, exact=exact, any_of=any_of)
        _assign_filter("file_type", self.file_types, exact=exact, any_of=any_of)
        _assign_filter("section", self.sections, exact=exact, any_of=any_of)

        payload_filter = PayloadFilter(exact=exact, any_of=any_of)
        return None if payload_filter.is_empty() else payload_filter

    def matches_payload(self, payload: dict[str, Any]) -> bool:
        """Return whether chunk payload satisfies these filters."""
        if self.is_empty():
            return True

        document_id = str(payload.get("document_id", ""))
        if self.document_ids and document_id not in self.document_ids:
            return False

        filename = str(payload.get("filename", ""))
        if self.filenames and filename not in self.filenames:
            return False

        file_type = str(payload.get("file_type", payload.get("content_type", "")))
        if self.file_types and file_type not in self.file_types:
            return False

        section = payload.get("section")
        section_value = str(section) if section not in (None, "") else ""
        return not (self.sections and section_value not in self.sections)


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _assign_filter(
    field_name: str,
    values: tuple[str, ...],
    *,
    exact: dict[str, str | int],
    any_of: dict[str, tuple[str, ...]],
) -> None:
    if not values:
        return
    if len(values) == 1:
        exact[field_name] = values[0]
        return
    any_of[field_name] = values
