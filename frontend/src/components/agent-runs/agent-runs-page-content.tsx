"use client";

import { useEffect, useState } from "react";
import { Info } from "lucide-react";
import { AgentRunList } from "@/components/agent-runs/agent-run-list";
import { useAgentRuns } from "@/lib/hooks/use-agent-runs";

const PAGE_SIZE = 20;

export function AgentRunsPageContent() {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "success" | "failure">("all");
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search), 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    setOffset(0);
  }, [debouncedSearch, statusFilter]);

  const { data, isLoading, isError, isFetching } = useAgentRuns({
    search: debouncedSearch || undefined,
    status: statusFilter === "all" ? undefined : statusFilter,
    limit: PAGE_SIZE,
    offset,
  });

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6">
      <div
        role="note"
        className="flex gap-3 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600"
      >
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden />
        <p>
          Agent runs are persisted server-side when you use{" "}
          <code className="rounded bg-slate-100 px-1">POST /agent/query</code>. History is available
          via <code className="rounded bg-slate-100 px-1">GET /agent/runs</code>.
        </p>
      </div>

      <AgentRunList
        runs={data?.runs}
        total={data?.total ?? 0}
        limit={data?.limit ?? PAGE_SIZE}
        offset={data?.offset ?? offset}
        isLoading={isLoading}
        isError={isError}
        search={search}
        statusFilter={statusFilter}
        onSearchChange={setSearch}
        onStatusFilterChange={setStatusFilter}
        onPageChange={setOffset}
      />

      {isFetching && !isLoading ? (
        <p className="text-center text-xs text-slate-400">Refreshing...</p>
      ) : null}
    </div>
  );
}
