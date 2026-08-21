"""Document ingestion API schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    """Response returned after a successful document ingestion."""

    document_id: str
    filename: str
    content_type: str
    file_type: str
    file_size: int = Field(ge=0)
    checksum: str
    source: str
    page_count: int = Field(ge=1)
    pages_stored: int = Field(ge=1)
    chunks_stored: int = Field(ge=1)
    status: Literal["ingested"] = "ingested"


class DocumentBatchIngestItem(BaseModel):
    """Single document result within a batch upload response."""

    document_id: str
    filename: str
    content_type: str
    file_type: str
    file_size: int = Field(ge=0)
    checksum: str
    source: str
    page_count: int = Field(ge=1)
    pages_stored: int = Field(ge=1)
    chunks_stored: int = Field(ge=1)
    status: Literal["ingested"] = "ingested"


class DocumentBatchIngestResponse(BaseModel):
    """Response returned after ingesting multiple documents."""

    documents: list[DocumentBatchIngestItem]
    total_documents: int = Field(ge=1)
    status: Literal["ingested"] = "ingested"


class DocumentDeleteResponse(BaseModel):
    """Response returned after deleting an ingested document."""

    document_id: str
    chunks_deleted: int = Field(ge=0)
    checksum: str | None = None
    filename: str | None = None
    status: Literal["deleted"] = "deleted"
