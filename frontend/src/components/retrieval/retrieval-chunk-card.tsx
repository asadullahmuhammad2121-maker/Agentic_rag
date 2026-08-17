"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";
import type { RetrievedChunkResult } from "@/lib/types/retrieval";
import { formatRetrievalMethod, formatScore } from "@/lib/retrieval/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface RetrievalChunkCardProps {
  chunk: RetrievedChunkResult;
  rank: number;
}

export function RetrievalChunkCard({ chunk, rank }: RetrievalChunkCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-3 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">#{rank}</Badge>
              <span className="flex items-center gap-1.5 font-medium text-slate-900">
                <FileText className="h-4 w-4 shrink-0 text-slate-400" />
                {chunk.filename}
              </span>
              <Badge variant="secondary">{formatRetrievalMethod(chunk.retrieval_method)}</Badge>
            </div>
            <dl className="grid gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-2 lg:grid-cols-4">
              <div>
                <dt className="inline">Page: </dt>
                <dd className="inline">{chunk.page_number}</dd>
              </div>
              <div>
                <dt className="inline">Section: </dt>
                <dd className="inline">{chunk.section ?? "—"}</dd>
              </div>
              <div>
                <dt className="inline">Chunk ID: </dt>
                <dd className="inline break-all">{chunk.chunk_id}</dd>
              </div>
              <div>
                <dt className="inline">Score: </dt>
                <dd className="inline font-medium text-slate-700">{formatScore(chunk.score)}</dd>
              </div>
            </dl>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="shrink-0"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? (
              <>
                <ChevronUp className="h-4 w-4" />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4" />
                Expand
              </>
            )}
          </Button>
        </div>
        <p className={cn("text-sm text-slate-700", !expanded && "line-clamp-2")}>{chunk.text}</p>
      </CardHeader>
      {expanded ? (
        <CardContent className="border-t border-slate-100 bg-slate-50 p-4 text-sm text-slate-700">
          <dl className="grid gap-3 sm:grid-cols-2">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Document ID
              </dt>
              <dd className="mt-1 break-all">{chunk.document_id}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">Source</dt>
              <dd className="mt-1 break-all">{chunk.source}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Chunk index
              </dt>
              <dd className="mt-1">{chunk.chunk_index}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Chunking strategy
              </dt>
              <dd className="mt-1">{chunk.chunking_strategy}</dd>
            </div>
          </dl>
          <div className="mt-4">
            <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Full chunk text
            </dt>
            <dd className="mt-2 whitespace-pre-wrap rounded-md border border-slate-200 bg-white p-3">
              {chunk.text}
            </dd>
          </div>
        </CardContent>
      ) : null}
    </Card>
  );
}
