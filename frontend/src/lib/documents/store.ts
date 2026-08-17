import { DOCUMENTS_STORAGE_KEY } from "@/lib/constants/documents";
import type { DocumentIngestResponse, StoredDocument } from "@/lib/types/documents";

function isStoredDocument(value: unknown): value is StoredDocument {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.document_id === "string" &&
    typeof record.filename === "string" &&
    typeof record.uploaded_at === "string" &&
    record.status === "ingested"
  );
}

function readRaw(): StoredDocument[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(DOCUMENTS_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isStoredDocument);
  } catch {
    return [];
  }
}

function writeRaw(documents: StoredDocument[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(DOCUMENTS_STORAGE_KEY, JSON.stringify(documents));
}

export function listStoredDocuments(): StoredDocument[] {
  return readRaw().sort(
    (a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime(),
  );
}

export function getStoredDocument(documentId: string): StoredDocument | undefined {
  return readRaw().find((doc) => doc.document_id === documentId);
}

export function upsertStoredDocuments(ingested: DocumentIngestResponse[]): StoredDocument[] {
  const existing = readRaw();
  const byId = new Map(existing.map((doc) => [doc.document_id, doc]));
  const uploadedAt = new Date().toISOString();

  for (const doc of ingested) {
    const previous = byId.get(doc.document_id);
    byId.set(doc.document_id, {
      ...doc,
      uploaded_at: previous?.uploaded_at ?? uploadedAt,
    });
  }

  const merged = [...byId.values()].sort(
    (a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime(),
  );
  writeRaw(merged);
  return merged;
}

export function removeStoredDocument(documentId: string): StoredDocument[] {
  const next = readRaw().filter((doc) => doc.document_id !== documentId);
  writeRaw(next);
  return next;
}
