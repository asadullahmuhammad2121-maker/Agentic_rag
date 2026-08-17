"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { CitationResponse } from "@/lib/types/agent";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ExternalLink, FileText, Globe } from "lucide-react";

interface CitationListProps {
  citations: CitationResponse[];
}

function CitationIcon({ fileType }: { fileType: string }) {
  if (fileType === "web") {
    return <Globe className="h-4 w-4 text-blue-600" />;
  }
  return <FileText className="h-4 w-4 text-slate-600" />;
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-900">
        Sources ({citations.length})
      </h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {citations.map((citation) => (
          <Card key={`${citation.chunk_id}-${citation.label}`} className="shadow-none">
            <CardHeader className="flex flex-row items-start gap-3 space-y-0 p-4 pb-2">
              <CitationIcon fileType={citation.file_type} />
              <div className="min-w-0 flex-1">
                <CardTitle className="truncate text-sm font-medium">
                  {citation.filename || citation.source}
                </CardTitle>
                <div className="mt-1 flex flex-wrap gap-1">
                  <Badge variant="outline">{citation.label}</Badge>
                  <Badge variant="secondary">{citation.file_type}</Badge>
                  <Badge variant="secondary">score {citation.score.toFixed(2)}</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-1 p-4 pt-0 text-xs text-slate-500">
              {citation.file_type === "web" ? (
                <a
                  href={citation.source}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-blue-600 hover:underline"
                >
                  {citation.source}
                  <ExternalLink className="h-3 w-3" />
                </a>
              ) : (
                <>
                  <p>Page {citation.page_number}</p>
                  {citation.section ? <p>Section: {citation.section}</p> : null}
                  <p className="truncate">Chunk: {citation.chunk_id}</p>
                </>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

interface MarkdownAnswerProps {
  content: string;
}

export function MarkdownAnswer({ content }: MarkdownAnswerProps) {
  return (
    <div className="prose prose-slate max-w-none prose-headings:font-semibold prose-p:leading-relaxed prose-pre:bg-slate-900 prose-pre:text-slate-100">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
