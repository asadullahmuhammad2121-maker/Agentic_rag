"""Document upload routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from app.api.deps import IngestionServiceDep
from app.core.logging import get_logger
from app.schemas.documents import DocumentIngestResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

PdfUpload = Annotated[UploadFile, File(description="PDF file to ingest")]


@router.post(
    "/upload",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(
    ingestion_service: IngestionServiceDep,
    file: PdfUpload,
) -> DocumentIngestResponse:
    """Validate and ingest a PDF into the vector store."""
    content = await file.read()
    filename = file.filename or "upload.pdf"
    content_type = file.content_type

    logger.info(
        "document_upload_received",
        extra={
            "operation": "upload_document",
            "document_filename": filename,
            "content_type": content_type,
            "file_size": len(content),
        },
    )

    result = ingestion_service.ingest_pdf(
        filename=filename,
        content=content,
        content_type=content_type,
    )
    return DocumentIngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        content_type=result.content_type,
        file_size=result.file_size,
        checksum=result.checksum,
        page_count=result.page_count,
        pages_stored=result.pages_stored,
        chunks_stored=result.chunks_stored,
        status="ingested",
    )
