"use client";

import { Loader2, Search } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { useRetrievalExplore } from "@/lib/hooks/use-retrieval";
import { ApiError } from "@/lib/api/client";
import { RetrievalPipeline } from "@/components/retrieval/retrieval-pipeline";
import { RetrievalResultsPanel } from "@/components/retrieval/retrieval-results-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";

export function RetrievalExplorerContent() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState<number | undefined>(undefined);
  const mutation = useRetrievalExplore();

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      toast.error("Please enter a query");
      return;
    }
    mutation.mutate({
      query: trimmed,
      ...(topK !== undefined ? { top_k: topK } : {}),
    });
  };

  const response = mutation.data;

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Search className="h-4 w-4" />
            Retrieval query
          </CardTitle>
          <CardDescription>
            Runs the configured backend retrieval pipeline via{" "}
            <code className="rounded bg-slate-100 px-1">POST /retrieval/explore</code> without LLM
            answer generation.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Enter a search query to inspect retrieval stages and chunk results..."
              rows={3}
              disabled={mutation.isPending}
              aria-label="Retrieval query"
            />
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <label className="flex flex-col gap-1 text-sm text-slate-600">
                Top K (optional)
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={topK ?? ""}
                  onChange={(event) => {
                    const value = event.target.value;
                    setTopK(value ? Number(value) : undefined);
                  }}
                  placeholder="Default from backend"
                  disabled={mutation.isPending}
                  className="w-full sm:w-40"
                />
              </label>
              <Button type="submit" disabled={mutation.isPending || !query.trim()}>
                {mutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running retrieval...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Explore retrieval
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {mutation.isPending ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Retrieval pipeline</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </CardContent>
        </Card>
      ) : null}

      {mutation.isError && !response ? (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-sm text-red-700">
            {mutation.error instanceof ApiError
              ? mutation.error.message
              : "Something went wrong while running retrieval."}
          </CardContent>
        </Card>
      ) : null}

      {!mutation.isPending && !response && !mutation.isError ? (
        <Card className="border-dashed">
          <CardContent className="p-10 text-center text-slate-500">
            <Search className="mx-auto mb-3 h-8 w-8 text-slate-300" />
            <p className="font-medium text-slate-700">No retrieval run yet</p>
            <p className="mt-1 text-sm">
              Submit a query to inspect pipeline stages and retrieved chunks.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {response ? (
        <>
          <Card>
            <CardHeader className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="text-base">Retrieval pipeline</CardTitle>
                {response.configuration.hybrid_search_enabled ? (
                  <Badge variant="success">Hybrid</Badge>
                ) : (
                  <Badge variant="outline">Vector only</Badge>
                )}
                {response.configuration.multi_query_enabled ? (
                  <Badge variant="outline">Multi-query</Badge>
                ) : null}
                {response.configuration.query_transformation_enabled ? (
                  <Badge variant="outline">Query transform</Badge>
                ) : null}
                {response.configuration.context_optimization_enabled ? (
                  <Badge variant="outline">Context opt</Badge>
                ) : null}
              </div>
              <CardDescription>
                Query: &ldquo;{response.query}&rdquo;
                {response.retrieval_query !== response.query ? (
                  <>
                    {" "}
                    · Retrieval query: &ldquo;{response.retrieval_query}&rdquo;
                  </>
                ) : null}
              </CardDescription>
              {response.generated_queries && response.generated_queries.length > 0 ? (
                <div className="rounded-lg border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
                  <p className="font-medium text-slate-700">Generated queries</p>
                  <ul className="mt-2 list-inside list-disc space-y-1">
                    {response.generated_queries.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </CardHeader>
            <CardContent>
              <RetrievalPipeline
                pipeline={response.pipeline}
                configuration={response.configuration}
              />
            </CardContent>
          </Card>

          {response.results.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="p-10 text-center text-sm text-slate-500">
                <p className="font-medium text-slate-700">No chunks retrieved</p>
                <p className="mt-1">
                  The pipeline completed but returned zero chunks. Try a different query or upload
                  documents first.
                </p>
              </CardContent>
            </Card>
          ) : (
            <RetrievalResultsPanel response={response} />
          )}
        </>
      ) : null}
    </div>
  );
}
