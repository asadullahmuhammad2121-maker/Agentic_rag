"""Document upload routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import IngestionServiceDep, SettingsDep
from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.schemas.documents import (
    DocumentBatchIngestItem,
    DocumentBatchIngestResponse,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummaryResponse,
)
from app.services.ingestion.service import DeletedDocument, IngestedDocument, ListedDocument
from app.utils.filenames import sanitize_upload_filename

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

DocumentUpload = Annotated[UploadFile, File(description="Document file to ingest")]
OptionalDocumentUpload = Annotated[
    UploadFile | None,
    File(description="Single document file to ingest (backward compatible)"),
]
AdditionalDocumentUploads = Annotated[
    list[UploadFile] | None,
    File(description="Optional additional document files to ingest in one request"),
]


def _to_ingest_response(result: IngestedDocument) -> DocumentIngestResponse:
    return DocumentIngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        content_type=result.content_type,
        file_type=result.file_type,
        file_size=result.file_size,
        checksum=result.checksum,
        source=result.source,
        page_count=result.page_count,
        pages_stored=result.pages_stored,
        chunks_stored=result.chunks_stored,
        status="ingested",
    )


def _to_batch_item(result: IngestedDocument) -> DocumentBatchIngestItem:
    return DocumentBatchIngestItem(
        document_id=result.document_id,
        filename=result.filename,
        content_type=result.content_type,
        file_type=result.file_type,
        file_size=result.file_size,
        checksum=result.checksum,
        source=result.source,
        page_count=result.page_count,
        pages_stored=result.pages_stored,
        chunks_stored=result.chunks_stored,
        status="ingested",
    )


def _to_delete_response(result: DeletedDocument) -> DocumentDeleteResponse:
    return DocumentDeleteResponse(
        document_id=result.document_id,
        chunks_deleted=result.chunks_deleted,
        checksum=result.checksum,
        filename=result.filename,
        status="deleted",
    )


def _to_summary_response(result: ListedDocument) -> DocumentSummaryResponse:
    return DocumentSummaryResponse(
        document_id=result.document_id,
        filename=result.filename,
        content_type=result.content_type,
        file_type=result.file_type,
        file_size=result.file_size,
        checksum=result.checksum,
        source=result.source,
        page_count=result.page_count,
        pages_stored=result.pages_stored,
        chunks_stored=result.chunks_stored,
        ingested_at=result.ingested_at,
        status="ingested",
    )


def _validate_upload_size(content: bytes, *, max_bytes: int) -> None:
    if len(content) > max_bytes:
        raise InvalidDocumentError(
            "File exceeds maximum upload size",
            details={
                "reason": "file_too_large",
                "file_size": len(content),
                "max_upload_file_size_bytes": max_bytes,
            },
        )


@router.post(
    "/upload",
    response_model=DocumentIngestResponse | DocumentBatchIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest one or more documents",
)
async def upload_document(
    ingestion_service: IngestionServiceDep,
    settings: SettingsDep,
    file: OptionalDocumentUpload = None,
    files: AdditionalDocumentUploads = None,
) -> DocumentIngestResponse | DocumentBatchIngestResponse:
    """Validate and ingest supported documents into the vector store."""
    uploads: list[UploadFile] = []
    if file is not None:
        uploads.append(file)
    if files:
        uploads.extend(files)

    if not uploads:
        raise InvalidDocumentError(
            "At least one file must be uploaded",
            details={"reason": "missing_file"},
        )

    if len(uploads) > settings.max_batch_upload_files:
        raise InvalidDocumentError(
            "Too many files in one upload request",
            details={
                "reason": "batch_limit_exceeded",
                "file_count": len(uploads),
                "max_batch_upload_files": settings.max_batch_upload_files,
            },
        )

    max_bytes = settings.max_upload_file_size_bytes

    if len(uploads) == 1:
        upload = uploads[0]
        content = await upload.read()
        _validate_upload_size(content, max_bytes=max_bytes)
        filename = sanitize_upload_filename(upload.filename or "upload.bin")
        logger.info(
            "document_upload_received",
            extra={
                "operation": "upload_document",
                "document_filename": filename,
                "content_type": upload.content_type,
                "file_size": len(content),
            },
        )
        result = ingestion_service.ingest_document(
            filename=filename,
            content=content,
            content_type=upload.content_type,
        )
        return _to_ingest_response(result)

    prepared: list[tuple[str, bytes, str | None]] = []
    for upload in uploads:
        content = await upload.read()
        _validate_upload_size(content, max_bytes=max_bytes)
        prepared.append(
            (
                sanitize_upload_filename(upload.filename or "upload.bin"),
                content,
                upload.content_type,
            )
        )

    logger.info(
        "document_batch_upload_received",
        extra={
            "operation": "upload_document",
            "document_count": len(prepared),
        },
    )
    results = ingestion_service.ingest_documents(prepared)
    return DocumentBatchIngestResponse(
        documents=[_to_batch_item(result) for result in results],
        total_documents=len(results),
        status="ingested",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents",
)
async def list_documents(
    ingestion_service: IngestionServiceDep,
) -> DocumentListResponse:
    """Return unique documents aggregated from vector-store chunk metadata."""
    documents = ingestion_service.list_documents()
    return DocumentListResponse(
        documents=[_to_summary_response(document) for document in documents],
        total_documents=len(documents),
        status="ok",
    )


@router.get(
    "/{document_id}",
    response_model=DocumentSummaryResponse,
    summary="Get ingested document metadata",
)
async def get_document(
    document_id: str,
    ingestion_service: IngestionServiceDep,
) -> DocumentSummaryResponse:
    """Return aggregated metadata for one ingested document."""
    result = ingestion_service.get_document(document_id)
    return _to_summary_response(result)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an ingested document",
)
async def delete_document(
    document_id: str,
    ingestion_service: IngestionServiceDep,
) -> DocumentDeleteResponse:
    """Remove all vector chunks and deduplication state for a document."""
    logger.info(
        "document_delete_received",
        extra={
            "operation": "delete_document",
            "document_id": document_id,
        },
    )
    result = ingestion_service.delete_document(document_id)
    return _to_delete_response(result)
