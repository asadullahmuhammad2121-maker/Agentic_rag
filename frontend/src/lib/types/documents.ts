/** Matches `DocumentIngestResponse` from the backend. */
export interface DocumentIngestResponse {
  document_id: string;
  filename: string;
  content_type: string;
  file_type: string;
  file_size: number;
  checksum: string;
  source: string;
  page_count: number;
  pages_stored: number;
  chunks_stored: number;
  status: "ingested";
}

/** Matches `DocumentSummaryResponse` from the backend. */
export interface DocumentSummary extends DocumentIngestResponse {
  ingested_at: string | null;
}

/** Matches `DocumentListResponse` from the backend. */
export interface DocumentListResponse {
  documents: DocumentSummary[];
  total_documents: number;
  status: "ok";
}

/** Matches `DocumentBatchIngestItem` from the backend. */
export type DocumentBatchIngestItem = DocumentIngestResponse;

/** Matches `DocumentBatchIngestResponse` from the backend. */
export interface DocumentBatchIngestResponse {
  documents: DocumentBatchIngestItem[];
  total_documents: number;
  status: "ingested";
}

export type DocumentUploadResponse = DocumentIngestResponse | DocumentBatchIngestResponse;

export function isBatchUploadResponse(
  response: DocumentUploadResponse,
): response is DocumentBatchIngestResponse {
  return "documents" in response;
}

export function uploadResponseToDocuments(
  response: DocumentUploadResponse,
): DocumentIngestResponse[] {
  if (isBatchUploadResponse(response)) {
    return response.documents;
  }
  return [response];
}
