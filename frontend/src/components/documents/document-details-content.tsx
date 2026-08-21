"use client";

import Link from "next/link";
import { ArrowLeft, FileText, Layers, LayoutList } from "lucide-react";
import { useDocumentDetail } from "@/lib/hooks/use-documents";
import {
  formatDateTime,
  formatFileSize,
  formatFileType,
} from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface DocumentDetailsContentProps {
  documentId: string;
}

function MetadataItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 break-all text-sm font-medium text-slate-900">{value}</dd>
    </div>
  );
}

export function DocumentDetailsContent({ documentId }: DocumentDetailsContentProps) {
  const { data: document, isLoading, isError, error } = useDocumentDetail(documentId);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <Skeleton className="h-10 w-40" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !document) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <Button variant="outline" size="sm" asChild>
          <Link href="/documents">
            <ArrowLeft className="h-4 w-4" />
            Back to documents
          </Link>
        </Button>
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-sm text-red-700">
            {error instanceof Error ? error.message : "This document was not found in the knowledge base."}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Button variant="outline" size="sm" asChild>
          <Link href="/documents">
            <ArrowLeft className="h-4 w-4" />
            Back to documents
          </Link>
        </Button>
        <div className="flex flex-wrap gap-2">
          <Badge variant="success">Ingested</Badge>
          <Badge variant="outline">{formatFileType(document.file_type)}</Badge>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-start gap-2 text-lg">
            <FileText className="mt-0.5 h-5 w-5 shrink-0 text-slate-500" />
            {document.filename}
          </CardTitle>
          <CardDescription>
            Ingested {document.ingested_at ? formatDateTime(document.ingested_at) : "—"} · ID{" "}
            {document.document_id}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <MetadataItem label="Document ID" value={document.document_id} />
            <MetadataItem label="Filename" value={document.filename} />
            <MetadataItem label="Source" value={document.source} />
            <MetadataItem label="Content type" value={document.content_type} />
            <MetadataItem label="File type" value={formatFileType(document.file_type)} />
            <MetadataItem label="File size" value={formatFileSize(document.file_size)} />
            <MetadataItem label="Checksum" value={document.checksum} />
            <MetadataItem label="Status" value={document.status} />
            <MetadataItem
              label="Ingested at"
              value={document.ingested_at ? formatDateTime(document.ingested_at) : "—"}
            />
          </dl>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Pages</CardDescription>
            <CardTitle className="text-2xl">{document.page_count}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-500">
            {document.pages_stored} stored during ingestion
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Chunks indexed</CardDescription>
            <CardTitle className="text-2xl">{document.chunks_stored}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-500">
            Total chunks written to the vector store
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Processing</CardDescription>
            <CardTitle className="text-2xl capitalize">{document.status}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-500">
            Upload API returns status after ingestion completes
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <LayoutList className="h-4 w-4" />
            Pages &amp; sections
          </CardTitle>
          <CardDescription>
            Aggregate counts from the document metadata API.
          </CardDescription>
        </CardHeader>
        <CardContent className="rounded-lg border border-dashed p-8 text-center text-sm text-slate-500">
          <Layers className="mx-auto mb-3 h-8 w-8 text-slate-300" />
          <p className="font-medium text-slate-700">Page-level detail unavailable</p>
          <p className="mt-1">
            The document API includes aggregate counts ({document.page_count} pages,{" "}
            {document.pages_stored} stored) but not individual page or section content.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Chunks</CardTitle>
          <CardDescription>
            Chunk text, IDs, page numbers, and sections are not exposed by the current API.
          </CardDescription>
        </CardHeader>
        <CardContent className="rounded-lg border border-dashed p-8 text-center text-sm text-slate-500">
          <p className="font-medium text-slate-700">No chunk listing endpoint</p>
          <p className="mt-1">
            {document.chunks_stored} chunks were indexed for this document. To browse chunk previews
            here, the backend would need to expose chunk metadata (e.g.{" "}
            <code className="rounded bg-slate-100 px-1">GET /documents/&#123;id&#125;/chunks</code>
            ).
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
