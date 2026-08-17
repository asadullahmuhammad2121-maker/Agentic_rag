"use client";

import Link from "next/link";
import { AlertCircle, RefreshCw } from "lucide-react";
import { useDashboardData } from "@/lib/hooks/use-dashboard";
import { formatRunStatus, formatRunTimestamp, formatToolsUsed } from "@/lib/agent-runs/utils";
import { getApiBaseUrl } from "@/lib/api/client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

function statusBadgeVariant(
  status: string,
): "success" | "warning" | "destructive" | "secondary" {
  if (status === "ok" || status === "alive" || status === "success") return "success";
  if (status === "degraded" || status === "partial") return "warning";
  if (status === "unavailable" || status === "failure" || status === "failed") {
    return "destructive";
  }
  return "secondary";
}

function componentLabel(name: string): string {
  switch (name) {
    case "qdrant":
      return "Qdrant";
    case "keyword_index":
      return "BM25 Index";
    default:
      return name;
  }
}

function StatusCard({
  label,
  status,
  detail,
}: {
  label: string;
  status: string;
  detail?: string | null;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-xl capitalize">{status}</CardTitle>
      </CardHeader>
      {detail ? (
        <CardContent>
          <p className="text-xs text-slate-500">{detail}</p>
        </CardContent>
      ) : null}
    </Card>
  );
}

function UnavailableCard({ label, reason }: { label: string; reason: string }) {
  return (
    <Card className="border-dashed">
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-xl text-slate-400">Unavailable</CardTitle>
      </CardHeader>
      <CardContent>
        <Badge variant="secondary">{reason}</Badge>
      </CardContent>
    </Card>
  );
}

export function DashboardContent() {
  const {
    health,
    healthError,
    ready,
    readyReachable,
    readyError,
    liveReachable,
    settings,
    settingsError,
    agentRuns,
    agentRunsError,
    isLoading,
    isError,
    refetch,
  } = useDashboardData();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex gap-3 text-sm text-red-700">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p className="font-medium">Backend unreachable</p>
              <p className="mt-1">
                Could not load dashboard data from{" "}
                <code className="rounded bg-red-100 px-1">{getApiBaseUrl()}</code>.
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const keywordComponent = health?.components.find((c) => c.name === "keyword_index");
  const qdrantComponent = health?.components.find((c) => c.name === "qdrant");
  const bm25Chunks =
    typeof keywordComponent?.metadata?.chunk_count === "number"
      ? keywordComponent.metadata.chunk_count
      : null;
  const hybridEnabled =
    settings?.rag.hybrid_search_enabled ??
    keywordComponent?.metadata?.hybrid_search_enabled === true;

  const ragTool = settings?.agent.tools.find((t) => t.name === "rag_retrieval");
  const tavilyTool = settings?.agent.tools.find((t) => t.name === "tavily_web_search");

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm text-slate-500">
            {health ? `${health.app} · ${health.environment} · v${health.version}` : null}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatusCard
          label="API status"
          status={healthError ? "unavailable" : (health?.status ?? "unknown")}
          detail={healthError ? "GET /health failed" : "From GET /health"}
        />
        <StatusCard
          label="Readiness"
          status={
            readyError ? "unavailable" : readyReachable ? (ready?.status ?? "ok") : "unavailable"
          }
          detail={readyError ? "GET /ready failed" : "From GET /ready"}
        />
        <StatusCard
          label="Liveness"
          status={liveReachable ? "alive" : "unavailable"}
          detail="From GET /live"
        />
        <StatusCard
          label="Qdrant"
          status={qdrantComponent?.status ?? "unknown"}
          detail={qdrantComponent?.detail}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>BM25 keyword search</CardDescription>
            <CardTitle className="text-xl capitalize">
              {keywordComponent?.status ?? "unknown"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Badge variant={hybridEnabled ? "success" : "secondary"}>
              Hybrid {hybridEnabled ? "enabled" : "disabled"}
            </Badge>
            {bm25Chunks !== null ? (
              <p className="text-xs text-slate-500">{bm25Chunks} indexed chunks</p>
            ) : null}
            {keywordComponent?.detail ? (
              <p className="text-xs text-slate-500">{keywordComponent.detail}</p>
            ) : null}
          </CardContent>
        </Card>

        {settings ? (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>RAG pipeline</CardDescription>
                <CardTitle className="text-xl">
                  {settings.rag.hybrid_search_enabled ? "Hybrid" : "Vector only"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-xs text-slate-500">
                <p>Top-K: {settings.rag.retrieval_top_k}</p>
                <p>Chunking: {settings.rag.chunking_strategy}</p>
                <p>
                  Multi-query: {settings.rag.multi_query_enabled ? "on" : "off"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Agent</CardDescription>
                <CardTitle className="text-xl">
                  {settings.agent.agent_enabled ? "Active" : "Inactive"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-xs text-slate-500">
                <p>Max steps: {settings.agent.agent_max_steps}</p>
                <p>Routing: {settings.agent.agent_routing_enabled ? "on" : "off"}</p>
                <p>Planning: {settings.agent.agent_planning_enabled ? "on" : "off"}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Web search (Tavily)</CardDescription>
                <CardTitle className="text-xl">
                  {settings.search.web_search_configured ? "Configured" : "Not configured"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Badge variant={settings.search.web_search_enabled ? "success" : "secondary"}>
                  {settings.search.web_search_enabled ? "Enabled" : "Disabled"}
                </Badge>
                <p className="text-xs text-slate-500">
                  Max results: {settings.search.tavily_max_results}
                </p>
              </CardContent>
            </Card>
          </>
        ) : (
          <>
            <UnavailableCard label="RAG pipeline" reason="GET /settings unavailable" />
            <UnavailableCard label="Agent" reason="GET /settings unavailable" />
            <UnavailableCard label="Web search" reason="GET /settings unavailable" />
          </>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Dependency health</CardTitle>
            <CardDescription>Component checks from GET /health</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {health?.components.length ? (
              health.components.map((component) => (
                <div
                  key={component.name}
                  className="flex items-start justify-between gap-3 rounded-lg border border-slate-200 px-3 py-2"
                >
                  <div>
                    <span className="text-sm font-medium">
                      {componentLabel(component.name)}
                    </span>
                    {component.detail ? (
                      <p className="mt-0.5 text-xs text-slate-500">{component.detail}</p>
                    ) : null}
                  </div>
                  <Badge variant={statusBadgeVariant(component.status)}>{component.status}</Badge>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No component data available.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Registered tools</CardTitle>
            <CardDescription>From GET /settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {settingsError || !settings ? (
              <p className="text-sm text-slate-500">Tool status unavailable.</p>
            ) : (
              settings.agent.tools.map((tool) => (
                <div
                  key={tool.name}
                  className="flex items-center justify-between rounded-lg border border-slate-200 px-3 py-2"
                >
                  <div>
                    <p className="text-sm font-medium">{tool.label}</p>
                    <p className="text-xs text-slate-500">{tool.name}</p>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <Badge variant={tool.available ? "success" : "secondary"}>
                      {tool.available ? "Available" : "Unavailable"}
                    </Badge>
                    <Badge variant={tool.configured ? "outline" : "secondary"}>
                      {tool.configured ? "Configured" : "Not configured"}
                    </Badge>
                  </div>
                </div>
              ))
            )}
            {ragTool ? (
              <p className="text-xs text-slate-400">
                RAG retrieval uses hybrid search when enabled ({hybridEnabled ? "yes" : "no"}).
              </p>
            ) : null}
            {tavilyTool && !tavilyTool.configured ? (
              <p className="text-xs text-slate-400">
                Tavily requires TAVILY_ENABLED and a server-side API key.
              </p>
            ) : null}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Index statistics</CardTitle>
            <CardDescription>Available metrics from backend health/settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between border-b border-slate-100 py-2">
              <span className="text-slate-600">BM25 indexed chunks</span>
              <span className="font-medium">
                {bm25Chunks !== null ? bm25Chunks : "—"}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-100 py-2">
              <span className="text-slate-600">Total documents</span>
              <Badge variant="secondary">Not exposed by API</Badge>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-600">Total queries</span>
              <Badge variant="secondary">Not exposed by API</Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle>Recent agent runs</CardTitle>
              <CardDescription>From GET /agent/runs</CardDescription>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/agent-runs">View all</Link>
            </Button>
          </CardHeader>
          <CardContent>
            {agentRunsError ? (
              <p className="text-sm text-slate-500">Could not load agent run history.</p>
            ) : !agentRuns?.runs.length ? (
              <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
                No agent runs yet. Submit a query in Agent Chat.
              </div>
            ) : (
              <ul className="divide-y divide-slate-100">
                {agentRuns.runs.map((run) => (
                  <li key={run.run_id} className="py-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <Link
                          href={`/agent-runs/${run.run_id}`}
                          className="line-clamp-1 text-sm font-medium text-slate-900 hover:underline"
                        >
                          {run.query}
                        </Link>
                        <p className="mt-1 text-xs text-slate-500">
                          {formatRunTimestamp(run.started_at)}
                          {run.tool_used ? ` · ${formatToolsUsed(run.tool_used)}` : ""}
                        </p>
                      </div>
                      <Badge variant={statusBadgeVariant(run.status)}>
                        {formatRunStatus(run.status)}
                      </Badge>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
