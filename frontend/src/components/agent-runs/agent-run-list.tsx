"use client";

import Link from "next/link";
import { ChevronRight, History, Search } from "lucide-react";
import type { AgentRunSummary } from "@/lib/types/agent-runs";
import {
  formatDuration,
  formatRunStatus,
  formatRunTimestamp,
  formatToolsUsed,
} from "@/lib/agent-runs/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

interface AgentRunListProps {
  runs: AgentRunSummary[] | undefined;
  total: number;
  limit: number;
  offset: number;
  isLoading: boolean;
  isError: boolean;
  search: string;
  statusFilter: "all" | "success" | "failure";
  onSearchChange: (value: string) => void;
  onStatusFilterChange: (value: "all" | "success" | "failure") => void;
  onPageChange: (offset: number) => void;
}

function statusBadgeVariant(status: AgentRunSummary["status"]) {
  return status === "success" ? "success" : "destructive";
}

export function AgentRunList({
  runs,
  total,
  limit,
  offset,
  isLoading,
  isError,
  search,
  statusFilter,
  onSearchChange,
  onStatusFilterChange,
  onPageChange,
}: AgentRunListProps) {
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const canPrev = offset > 0;
  const canNext = offset + limit < total;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Run history</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 5 }).map((_, index) => (
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
          Could not load agent run history from the backend.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="space-y-4">
        <div>
          <CardTitle className="text-base">Run history</CardTitle>
          <CardDescription>{total} persisted run{total === 1 ? "" : "s"}</CardDescription>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              value={search}
              onChange={(event) => onSearchChange(event.target.value)}
              placeholder="Search queries..."
              className="pl-9"
              aria-label="Search agent runs"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(event) =>
              onStatusFilterChange(event.target.value as "all" | "success" | "failure")
            }
            aria-label="Filter by status"
            className="h-10 rounded-md border border-slate-200 bg-white px-3 text-sm"
          >
            <option value="all">All statuses</option>
            <option value="success">Success</option>
            <option value="failure">Failed</option>
          </select>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!runs?.length ? (
          <div className="rounded-lg border border-dashed p-10 text-center">
            <History className="mx-auto mb-3 h-8 w-8 text-slate-300" />
            <p className="font-medium text-slate-700">No agent runs yet</p>
            <p className="mt-1 text-sm text-slate-500">
              Runs appear here after you submit queries in Agent Chat.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-100">
            {runs.map((run) => (
              <li key={run.run_id} className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center">
                <div className="min-w-0 flex-1 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={statusBadgeVariant(run.status)}>
                      {formatRunStatus(run.status)}
                    </Badge>
                    {run.tool_used ? (
                      <Badge variant="outline">{formatToolsUsed(run.tool_used)}</Badge>
                    ) : null}
                  </div>
                  <p className="line-clamp-2 font-medium text-slate-900">{run.query}</p>
                  <dl className="grid gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-2 lg:grid-cols-4">
                    <div>
                      <dt className="inline">Started: </dt>
                      <dd className="inline">{formatRunTimestamp(run.started_at)}</dd>
                    </div>
                    <div>
                      <dt className="inline">Duration: </dt>
                      <dd className="inline">{formatDuration(run.duration_ms)}</dd>
                    </div>
                    <div>
                      <dt className="inline">Steps: </dt>
                      <dd className="inline">{run.step_count}</dd>
                    </div>
                    <div>
                      <dt className="inline">Citations: </dt>
                      <dd className="inline">{run.citation_count}</dd>
                    </div>
                  </dl>
                  {run.status === "failure" && run.error_message ? (
                    <p className="text-xs text-red-700">{run.error_message}</p>
                  ) : null}
                </div>
                <Button variant="outline" size="sm" asChild className="shrink-0">
                  <Link href={`/agent-runs/${run.run_id}`}>
                    View run
                    <ChevronRight className="h-4 w-4" />
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        )}

        {total > limit ? (
          <div className="flex items-center justify-between border-t border-slate-100 pt-4 text-sm">
            <span className="text-slate-500">
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!canPrev}
                onClick={() => onPageChange(Math.max(0, offset - limit))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={!canNext}
                onClick={() => onPageChange(offset + limit)}
              >
                Next
              </Button>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
