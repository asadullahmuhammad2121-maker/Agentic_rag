"use client";

import type { AgentRunDetail } from "@/lib/types/agent-runs";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface AgentRunTimelineProps {
  run: AgentRunDetail;
}

function TimelineNode({
  title,
  active,
  description,
}: {
  title: string;
  active: boolean;
  description?: string;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div
          className={cn(
            "mt-1 h-2.5 w-2.5 rounded-full",
            active ? "bg-emerald-500" : "bg-slate-300",
          )}
        />
        <div className="min-h-6 flex-1 border-l border-dashed border-slate-300" />
      </div>
      <div className="min-w-0 flex-1 pb-4">
        <p className={cn("text-sm", active ? "font-medium text-slate-900" : "text-slate-400")}>
          {title}
        </p>
        {description ? <p className="mt-1 text-xs text-slate-500">{description}</p> : null}
      </div>
    </div>
  );
}

export function AgentRunTimeline({ run }: AgentRunTimelineProps) {
  const hasPlanning =
    run.metadata.decomposed === true ||
    run.steps.some((step) => step.action.type === "execute_plan");
  const hasToolExecution = run.steps.some((step) =>
    ["call_tool", "call_tools", "execute_plan"].includes(step.action.type),
  );
  const hasObservation = run.steps.some((step) => step.observation !== null);
  const hasGeneration = run.metadata.generated === true;
  const hasAnswer = Boolean(run.answer) && run.status === "success";

  const toolNames =
    (run.metadata.tool_names as string[] | undefined) ??
    (run.tool_used ? run.tool_used.split("+") : []);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-slate-900">Execution timeline</h3>
        {run.status === "failure" ? <Badge variant="destructive">Failed</Badge> : null}
      </div>

      <TimelineNode title="Query" active description={run.query} />
      <TimelineNode
        title="Planning"
        active={hasPlanning || hasToolExecution}
        description={
          hasPlanning
            ? "Query routed or decomposed into planned tasks"
            : hasToolExecution
              ? "Tool routing completed"
              : undefined
        }
      />
      <TimelineNode
        title="Tool Execution"
        active={hasToolExecution}
        description={
          toolNames.length > 0 ? toolNames.join(", ") : undefined
        }
      />
      <TimelineNode
        title="Observation"
        active={hasObservation}
        description={
          hasObservation
            ? `${run.steps.filter((step) => step.observation).length} observation(s) recorded`
            : undefined
        }
      />
      <TimelineNode
        title="Generation"
        active={hasGeneration}
        description={hasGeneration ? "Answer generated from tool outputs" : undefined}
      />
      <div className="flex gap-3">
        <div
          className={cn(
            "mt-1 h-2.5 w-2.5 rounded-full",
            hasAnswer ? "bg-emerald-500" : run.status === "failure" ? "bg-red-500" : "bg-slate-300",
          )}
        />
        <div className="min-w-0 flex-1">
          <p
            className={cn(
              "text-sm",
              hasAnswer || run.status === "failure"
                ? "font-medium text-slate-900"
                : "text-slate-400",
            )}
          >
            Final Answer
          </p>
          {run.status === "failure" && run.error_message ? (
            <p className="mt-1 text-xs text-red-700">{run.error_message}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
