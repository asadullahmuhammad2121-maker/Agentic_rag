"""Unit tests for structured logging safety."""

from __future__ import annotations

import logging

from app.core.logging import SensitiveDataFilter, setup_logging


def test_sensitive_filter_redacts_keys() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.groq_api_key = "sk-should-not-appear"  # type: ignore[attr-defined]
    record.api_key = "another-secret"  # type: ignore[attr-defined]
    record.operation = "health_check"  # type: ignore[attr-defined]

    filt = SensitiveDataFilter()
    assert filt.filter(record) is True
    assert record.groq_api_key == "[REDACTED]"  # type: ignore[attr-defined]
    assert record.api_key == "[REDACTED]"  # type: ignore[attr-defined]
    assert record.operation == "health_check"  # type: ignore[attr-defined]


def test_setup_logging_idempotent() -> None:
    setup_logging(level="INFO", service_name="test-service")
    setup_logging(level="WARNING", service_name="test-service")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
