"""Provider-agnostic vector-store payload filter specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PayloadFilter:
    """Exact-match and any-of payload conditions for vector search."""

    exact: dict[str, str | int] = field(default_factory=dict)
    any_of: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.exact and not self.any_of

    @classmethod
    def from_legacy_dict(cls, conditions: dict[str, Any]) -> PayloadFilter:
        """Convert legacy single-value filter mappings."""
        exact: dict[str, str | int] = {}
        any_of: dict[str, tuple[str, ...]] = {}
        for key, value in conditions.items():
            if isinstance(value, list | tuple):
                cleaned = tuple(str(item).strip() for item in value if str(item).strip())
                if cleaned:
                    any_of[key] = cleaned
            elif isinstance(value, int):
                exact[key] = value
            elif isinstance(value, str) and value.strip():
                exact[key] = value.strip()
        return cls(exact=exact, any_of=any_of)
