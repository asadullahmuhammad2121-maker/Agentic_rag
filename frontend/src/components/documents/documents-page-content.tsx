"use client";

import { Info } from "lucide-react";
import { DocumentList } from "@/components/documents/document-list";
import { DocumentUploadZone } from "@/components/documents/document-upload-zone";
import { useDocumentsList } from "@/lib/hooks/use-documents";

export function DocumentsPageContent() {
  const { data, isLoading, isError } = useDocumentsList();

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6">
      <div
        role="note"
        className="flex gap-3 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600"
      >
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden />
        <p>
          Documents are loaded from <code className="rounded bg-slate-100 px-1">GET /documents</code>
          . Upload uses{" "}
          <code className="rounded bg-slate-100 px-1">POST /documents/upload</code> and delete uses{" "}
          <code className="rounded bg-slate-100 px-1">DELETE /documents/&#123;id&#125;</code>.
        </p>
      </div>

      <DocumentUploadZone />
      <DocumentList documents={data} isLoading={isLoading} isError={isError} />
    </div>
  );
}
