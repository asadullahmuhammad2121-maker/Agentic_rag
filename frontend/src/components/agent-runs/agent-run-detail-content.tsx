"use client";

import Link from "next/link";
import { ArrowLeft, Loader2 } from "lucide-react";
import { AgentTrace } from "@/components/agent-chat/agent-trace";
import { CitationList, MarkdownAnswer } from "@/components/agent-chat/citations";
import { AgentRunTimeline } from "@/components/agent-runs/agent-run-timeline";
import { useAgentRunDetail } from "@/lib/hooks/use-agent-runs";
import { agentRunDetailToTraceData } from "@/lib/types/agent-runs";
import {
  formatDuration,
  formatRunStatus,
  formatRunTimestamp,
  formatToolsUsed,
} from "@/lib/agent-runs/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface AgentRunDetailContentProps {
  runId: string;
}

export function AgentRunDetailContent({ runId }: AgentRunDetailContentProps) {
  const { data: run, isLoading, isError, error } = useAgentRunDetail(runId);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <Skeleton className="h-10 w-40" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !run) {
    return (
      <div className="mx-auto max-w-5xl space-y-4">
        <Button variant="outline" size="sm" asChild>
          <Link href="/agent-runs">
            <ArrowLeft className="h-4 w-4" />
            Back to runs
          </Link>
        </Button>
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-sm text-red-700">
            {error instanceof Error ? error.message : "Agent run not found."}
          </CardContent>
        </Card>
      </div>
    );
  }

  const traceData = agentRunDetailToTraceData(run);

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Button variant="outline" size="sm" asChild>
          <Link href="/agent-runs">
            <ArrowLeft className="h-4 w-4" />
            Back to runs
          </Link>
        </Button>
        <div className="flex flex-wrap gap-2">
          <Badge variant={run.status === "success" ? "success" : "destructive"}>
            {formatRunStatus(run.status)}
          </Badge>
          {run.tool_used ? <Badge variant="outline">{formatToolsUsed(run.tool_used)}</Badge> : null}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{run.query}</CardTitle>
          <CardDescription>
            Run {run.run_id} · Started {formatRunTimestamp(run.started_at)} · Duration{" "}
            {formatDuration(run.duration_ms)}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Steps</dt>
              <dd className="mt-1 font-medium">{run.step_count}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Citations</dt>
              <dd className="mt-1 font-medium">{run.citation_count}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Completed</dt>
              <dd className="mt-1 font-medium">{formatRunTimestamp(run.completed_at)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Tools</dt>
              <dd className="mt-1 font-medium">{formatToolsUsed(run.tool_used)}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <AgentRunTimeline run={run} />

      {run.status === "failure" ? (
        <Card className="border-red-200 bg-red-50">
          <CardHeader>
            <CardTitle className="text-base text-red-800">Run failed</CardTitle>
            <CardDescription className="text-red-700">
              {run.error_code ? `Error code: ${run.error_code}` : "An error occurred during execution"}
            </CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-red-700">
            {run.error_message ?? "No error message was recorded for this run."}
          </CardContent>
        </Card>
      ) : null}

      {run.answer ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Final answer</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <MarkdownAnswer content={run.answer} />
            <CitationList citations={run.citations} />
          </CardContent>
        </Card>
      ) : null}

      <AgentTrace response={traceData} />
    </div>
  );
}

export function AgentRunDetailLoading() {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <Loader2 className="h-4 w-4 animate-spin" />
      Loading run details...
    </div>
  );
}
