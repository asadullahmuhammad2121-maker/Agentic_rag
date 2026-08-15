"""Document ingestion API schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    """Response returned after a successful PDF ingestion."""

    document_id: str
    filename: str
    content_type: str
    file_size: int = Field(ge=0)
    checksum: str
    page_count: int = Field(ge=1)
    pages_stored: int = Field(ge=1)
    chunks_stored: int = Field(ge=1)
    status: Literal["ingested"] = "ingested"
