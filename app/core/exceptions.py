"""Application exception hierarchy and FastAPI error handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base application error with a safe client-facing message."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "app_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(AppError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="configuration_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class QdrantConnectionError(AppError):
    """Raised when Qdrant is unreachable or returns a connection failure."""

    def __init__(self, message: str = "Unable to connect to vector store") -> None:
        super().__init__(
            message,
            code="qdrant_connection_error",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class VectorStoreError(AppError):
    """Raised for vector store operational failures."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="vector_store_error",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details,
        )


class ProviderError(AppError):
    """Raised when an external LLM or embedding provider fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_details = {"provider": provider, **(details or {})}
        super().__init__(
            message,
            code="provider_error",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=safe_details,
        )


class InvalidDocumentError(AppError):
    """Raised when an uploaded document fails validation or parsing."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="invalid_document",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class DuplicateDocumentError(AppError):
    """Raised when a document with the same checksum already exists."""

    def __init__(
        self,
        message: str = "A document with this content already exists",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="duplicate_document",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class DocumentIngestionError(AppError):
    """Raised when document ingestion fails after validation."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="document_ingestion_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DocumentNotFoundError(AppError):
    """Raised when a requested document is not present in the vector store."""

    def __init__(
        self,
        message: str = "Document not found",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="document_not_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class QueryError(AppError):
    """Raised when a RAG query request is invalid."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="query_error",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class AgentError(AppError):
    """Raised when the agent orchestrator cannot complete a run."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            code="agent_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class ErrorResponse(BaseModel):
    """Safe API error payload — never includes secrets or stack traces."""

    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable safe message")
    details: dict[str, Any] = Field(default_factory=dict)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers that return safe responses."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        logger.error(
            "application_error",
            extra={
                "operation": "exception_handler",
                "error_code": exc.code,
                "status_code": exc.status_code,
                "error_message": exc.message,
            },
        )
        body = ErrorResponse(error=exc.code, message=exc.message, details=exc.details)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        # Expose only field locations and messages — never raw input bodies with secrets.
        safe_errors = [
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "msg": err.get("msg", "Invalid value"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "validation_error",
            extra={"operation": "exception_handler", "error_count": len(safe_errors)},
        )
        body = ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details={"errors": safe_errors},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=body.model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unexpected_error",
            extra={
                "operation": "exception_handler",
                "error_type": type(exc).__name__,
            },
        )
        body = ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            details={},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )
