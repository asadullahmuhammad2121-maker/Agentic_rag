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

/** Matches `DocumentBatchIngestItem` from the backend. */
export type DocumentBatchIngestItem = DocumentIngestResponse;

/** Matches `DocumentBatchIngestResponse` from the backend. */
export interface DocumentBatchIngestResponse {
  documents: DocumentBatchIngestItem[];
  total_documents: number;
  status: "ingested";
}

export type DocumentUploadResponse = DocumentIngestResponse | DocumentBatchIngestResponse;

/** Client-side record enriched with upload timestamp (not from API). */
export interface StoredDocument extends DocumentIngestResponse {
  uploaded_at: string;
}

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
