"use client";

import { useMemo, useState } from "react";
import { Filter, Search } from "lucide-react";
import type {
  RetrievedChunkResult,
  RetrievalExploreResponse,
  RetrievalMethodFilter,
  RetrievalResultView,
  RetrievalSortDirection,
  RetrievalSortField,
} from "@/lib/types/retrieval";
import { RetrievalChunkCard } from "@/components/retrieval/retrieval-chunk-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface RetrievalResultsPanelProps {
  response: RetrievalExploreResponse;
}

function chunksForView(response: RetrievalExploreResponse, view: RetrievalResultView) {
  switch (view) {
    case "vector":
      return response.vector_results;
    case "bm25":
      return response.bm25_results;
    case "fused":
      return response.fused_results ?? [];
    case "final":
    default:
      return response.results;
  }
}

function sortChunks(
  chunks: RetrievedChunkResult[],
  field: RetrievalSortField,
  direction: RetrievalSortDirection,
): RetrievedChunkResult[] {
  const sorted = [...chunks].sort((a, b) => {
    switch (field) {
      case "filename":
        return a.filename.localeCompare(b.filename);
      case "page_number":
        return a.page_number - b.page_number;
      case "score":
      default:
        return b.score - a.score;
    }
  });
  return direction === "asc" ? sorted.reverse() : sorted;
}

export function RetrievalResultsPanel({ response }: RetrievalResultsPanelProps) {
  const [view, setView] = useState<RetrievalResultView>("final");
  const [search, setSearch] = useState("");
  const [methodFilter, setMethodFilter] = useState<RetrievalMethodFilter>("all");
  const [sortField, setSortField] = useState<RetrievalSortField>("score");
  const [sortDirection, setSortDirection] = useState<RetrievalSortDirection>("desc");

  const viewOptions = useMemo(() => {
    const options: { id: RetrievalResultView; label: string; count: number }[] = [
      { id: "final", label: "Final results", count: response.results.length },
      { id: "vector", label: "Vector search", count: response.vector_results.length },
    ];
    if (response.configuration.hybrid_search_enabled) {
      options.push(
        { id: "bm25", label: "BM25", count: response.bm25_results.length },
        {
          id: "fused",
          label: "Hybrid fusion",
          count: response.fused_results?.length ?? 0,
        },
      );
    }
    return options;
  }, [response]);

  const filtered = useMemo(() => {
    const source = chunksForView(response, view);
    const query = search.trim().toLowerCase();
    return sortChunks(
      source.filter((chunk) => {
        const matchesMethod =
          methodFilter === "all" || chunk.retrieval_method === methodFilter;
        const matchesSearch =
          !query ||
          chunk.filename.toLowerCase().includes(query) ||
          chunk.text.toLowerCase().includes(query) ||
          chunk.chunk_id.toLowerCase().includes(query) ||
          (chunk.section?.toLowerCase().includes(query) ?? false);
        return matchesMethod && matchesSearch;
      }),
      sortField,
      sortDirection,
    );
  }, [response, view, search, methodFilter, sortField, sortDirection]);

  return (
    <Card>
      <CardHeader className="space-y-4">
        <div>
          <CardTitle className="text-base">Retrieved chunks</CardTitle>
          <CardDescription>
            {filtered.length} of {chunksForView(response, view).length} chunks shown
          </CardDescription>
        </div>

        <div className="flex flex-wrap gap-2">
          {viewOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => setView(option.id)}
              className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                view === option.id
                  ? "border-slate-900 bg-slate-900 text-white"
                  : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
              }`}
            >
              {option.label} ({option.count})
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-3 lg:flex-row">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search filename, text, chunk ID..."
              className="pl-9"
              aria-label="Search retrieved chunks"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <Filter className="h-4 w-4" />
              <span className="sr-only sm:not-sr-only">Method</span>
              <select
                value={methodFilter}
                onChange={(event) =>
                  setMethodFilter(event.target.value as RetrievalMethodFilter)
                }
                aria-label="Filter by retrieval method"
                className="h-10 rounded-md border border-slate-200 bg-white px-3 text-sm"
              >
                <option value="all">All methods</option>
                <option value="vector">Vector</option>
                <option value="bm25">BM25</option>
                <option value="hybrid_fusion">Hybrid fusion</option>
                <option value="multi_query">Multi-query</option>
              </select>
            </label>
            <select
              value={`${sortField}:${sortDirection}`}
              onChange={(event) => {
                const [field, direction] = event.target.value.split(":");
                setSortField(field as RetrievalSortField);
                setSortDirection(direction as RetrievalSortDirection);
              }}
              aria-label="Sort retrieved chunks"
              className="h-10 rounded-md border border-slate-200 bg-white px-3 text-sm"
            >
              <option value="score:desc">Score (high → low)</option>
              <option value="score:asc">Score (low → high)</option>
              <option value="filename:asc">Filename (A → Z)</option>
              <option value="page_number:asc">Page (low → high)</option>
            </select>
          </div>
        </div>

        {typeof response.metadata.intermediate_query_note === "string" ? (
          <p className="text-xs text-slate-500">{response.metadata.intermediate_query_note}</p>
        ) : null}
      </CardHeader>

      <CardContent className="space-y-4">
        {filtered.length === 0 ? (
          <div className="rounded-lg border border-dashed p-10 text-center text-sm text-slate-500">
            <p className="font-medium text-slate-700">No chunks match your filters</p>
            <p className="mt-1">Try another view, search term, or retrieval method filter.</p>
          </div>
        ) : (
          filtered.map((chunk, index) => (
            <RetrievalChunkCard key={`${view}-${chunk.chunk_id}`} chunk={chunk} rank={index + 1} />
          ))
        )}
      </CardContent>
    </Card>
  );
}
