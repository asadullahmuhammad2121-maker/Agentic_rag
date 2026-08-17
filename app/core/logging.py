"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

# Fields that must never appear in log records or extras.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "authorization",
        "groq_api_key",
        "huggingface_api_key",
        "tavily_api_key",
        "prompt",
        "system_prompt",
        "user_prompt",
        "document_content",
        "full_document",
        "text",
        "page_text",
        "extracted_text",
        "context",
    }
)


class SensitiveDataFilter(logging.Filter):
    """Redact known sensitive keys from log record extras."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key.lower() in _SENSITIVE_KEYS:
                setattr(record, key, "[REDACTED]")
        return True


class StructuredFormatter(logging.Formatter):
    """Emit key=value structured log lines suitable for aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"timestamp={self.formatTime(record, self.datefmt)} "
            f"level={record.levelname} "
            f"logger={record.name} "
            f"message={record.getMessage()}"
        )

        extras: list[str] = []
        skip = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key in skip or value is None:
                continue
            extras.append(f"{key}={_safe_repr(value)}")

        if extras:
            base = f"{base} {' '.join(extras)}"

        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


def _safe_repr(value: Any) -> str:
    text = str(value)
    # Avoid dumping large payloads into logs.
    if len(text) > 500:
        return f"{text[:500]}...<truncated>"
    return text.replace(" ", "_") if "\n" not in text else text.replace("\n", "\\n")


def setup_logging(level: str = "INFO", service_name: str = "rag-foundation") -> None:
    """Configure root and application loggers with structured formatting."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())
    handler.addFilter(SensitiveDataFilter())
    handler.setFormatter(
        StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"),
    )
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(service_name).info(
        "logging_configured",
        extra={"operation": "setup_logging", "service": service_name},
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
