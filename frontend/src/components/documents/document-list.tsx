"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ChevronRight,
  FileText,
  Search,
  Trash2,
} from "lucide-react";
import type { DocumentSummary } from "@/lib/types/documents";
import { formatDateTime, formatFileSize, formatFileType } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { DeleteDocumentDialog } from "@/components/documents/delete-document-dialog";

interface DocumentListProps {
  documents: DocumentSummary[] | undefined;
  isLoading: boolean;
  isError: boolean;
}

function statusBadge(status: DocumentSummary["status"]) {
  if (status === "ingested") {
    return <Badge variant="success">Ingested</Badge>;
  }
  return <Badge variant="secondary">{status}</Badge>;
}

export function DocumentList({ documents, isLoading, isError }: DocumentListProps) {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [deleteTarget, setDeleteTarget] = useState<DocumentSummary | null>(null);

  const fileTypes = useMemo(() => {
    if (!documents) return [];
    return [...new Set(documents.map((doc) => doc.file_type))].sort();
  }, [documents]);

  const filtered = useMemo(() => {
    if (!documents) return [];
    const query = search.trim().toLowerCase();
    return documents.filter((doc) => {
      const matchesSearch =
        !query ||
        doc.filename.toLowerCase().includes(query) ||
        doc.document_id.toLowerCase().includes(query) ||
        doc.source.toLowerCase().includes(query);
      const matchesType = typeFilter === "all" || doc.file_type === typeFilter;
      return matchesSearch && matchesType;
    });
  }, [documents, search, typeFilter]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Documents</CardTitle>
          <CardDescription>Loading your uploaded documents...</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-20 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent className="p-6 text-sm text-red-700">
          Could not load documents from the backend. Check that the API is running and try again.
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeader className="space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-base">Documents</CardTitle>
              <CardDescription>
                {documents?.length ?? 0} document{(documents?.length ?? 0) === 1 ? "" : "s"} in the
                knowledge base
              </CardDescription>
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search by filename or ID..."
                className="pl-9"
                aria-label="Search documents"
              />
            </div>
            <select
              value={typeFilter}
              onChange={(event) => setTypeFilter(event.target.value)}
              aria-label="Filter by file type"
              className="h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
            >
              <option value="all">All types</option>
              {fileTypes.map((type) => (
                <option key={type} value={type}>
                  {formatFileType(type)}
                </option>
              ))}
            </select>
          </div>
        </CardHeader>

        <CardContent>
          {!documents?.length ? (
            <div className="rounded-lg border border-dashed p-10 text-center">
              <FileText className="mx-auto mb-3 h-8 w-8 text-slate-300" />
              <p className="font-medium text-slate-700">No documents uploaded</p>
              <p className="mt-1 text-sm text-slate-500">
                Upload a file above to ingest it into the vector store. Documents appear here once
                ingestion succeeds.
              </p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-slate-500">
              No documents match your search or filter.
            </div>
          ) : (
            <ul className="divide-y divide-slate-100">
              {filtered.map((doc) => (
                <li key={doc.document_id} className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Link
                        href={`/documents/${doc.document_id}`}
                        className="truncate font-medium text-slate-900 hover:underline"
                      >
                        {doc.filename}
                      </Link>
                      {statusBadge(doc.status)}
                      <Badge variant="outline">{formatFileType(doc.file_type)}</Badge>
                    </div>
                    <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-2 lg:grid-cols-4">
                      <div>
                        <dt className="inline">Size: </dt>
                        <dd className="inline">{formatFileSize(doc.file_size)}</dd>
                      </div>
                      <div>
                        <dt className="inline">Pages: </dt>
                        <dd className="inline">{doc.page_count}</dd>
                      </div>
                      <div>
                        <dt className="inline">Chunks: </dt>
                        <dd className="inline">{doc.chunks_stored}</dd>
                      </div>
                      <div>
                        <dt className="inline">Ingested: </dt>
                        <dd className="inline">
                          {doc.ingested_at ? formatDateTime(doc.ingested_at) : "—"}
                        </dd>
                      </div>
                    </dl>
                  </div>

                  <div className="flex shrink-0 items-center gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`/documents/${doc.document_id}`}>
                        Details
                        <ChevronRight className="h-4 w-4" />
                      </Link>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:bg-red-50 hover:text-red-700"
                      aria-label={`Delete ${doc.filename}`}
                      onClick={() => setDeleteTarget(doc)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <DeleteDocumentDialog
        document={deleteTarget}
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      />
    </>
  );
}