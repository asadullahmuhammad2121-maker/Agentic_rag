"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import type { AgentQueryResponse, AgentStepResponse } from "@/lib/types/agent";
import { formatActionType, formatCalculatorResult, formatToolLabel } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";

interface AgentTraceProps {
  response: Pick<
    AgentQueryResponse,
    "answer" | "citations" | "tool_used" | "steps" | "metadata"
  >;
}

function toolDisplayName(toolName: string): string {
  if (toolName === "rag_retrieval") return "RAG Retrieval";
  if (toolName === "tavily_web_search") return "Tavily Web Search";
  if (toolName === "calculator") return "Calculator";
  return toolName;
}

function renderStep(step: AgentStepResponse, index: number) {
  const { action, observation } = step;

  return (
    <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">Step {index + 1}</Badge>
        <Badge>{formatActionType(action.type)}</Badge>
        {action.tool_name ? (
          <Badge variant="outline">{toolDisplayName(action.tool_name)}</Badge>
        ) : null}
        {action.tool_names.length > 0
          ? action.tool_names.map((name) => (
              <Badge key={name} variant="outline">
                {toolDisplayName(name)}
              </Badge>
            ))
          : null}
      </div>
      {action.reasoning ? (
        <p className="mt-2 text-sm text-slate-600">{action.reasoning}</p>
      ) : null}
      {observation ? (
        <div className="mt-3 text-sm text-slate-600">
          <p>
            Observation: {toolDisplayName(observation.tool_name)} —{" "}
            {observation.success ? "success" : "failed"}
            {observation.tool_name !== "calculator" ? (
              <> · {observation.citation_count} citation(s)</>
            ) : null}
          </p>
          {observation.tool_name === "calculator" &&
          observation.success &&
          observation.expression ? (
            <p className="mt-1 font-mono text-xs text-slate-700">
              {observation.expression} = {formatCalculatorResult(observation.result ?? null)}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export function AgentTrace({ response }: AgentTraceProps) {
  const { steps, metadata, tool_used } = response;
  const hasSteps = steps.length > 0;
  const hasPlanning =
    metadata.decomposed === true ||
    steps.some((s) => s.action.type === "execute_plan");
  const hasGeneration = metadata.generated === true;
  const toolNames =
    (metadata.tool_names as string[] | undefined) ??
    (tool_used ? tool_used.split("+") : []);

  return (
    <Collapsible defaultOpen={false}>
      <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3 text-left text-sm font-medium hover:bg-slate-50">
        <span>Agent Trace</span>
        <ChevronDown className="h-4 w-4 transition-transform [[data-state=open]_&]:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 space-y-4 rounded-lg border border-slate-200 bg-white p-4">
        <TraceNode title="User Query" active />
        <TraceConnector />
        <TraceNode title="Agent Planning / Routing" active={hasPlanning || hasSteps} />
        <TraceConnector />
        <div className="space-y-2 pl-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Tool Execution
          </p>
          {toolNames.length > 0 ? (
            toolNames.map((name) => (
              <div key={name} className="flex items-center gap-2 text-sm text-slate-700">
                <ChevronRight className="h-4 w-4 text-slate-400" />
                {toolDisplayName(name)}
              </div>
            ))
          ) : (
            <p className="text-sm text-slate-400">No tools recorded</p>
          )}
        </div>
        <TraceConnector />
        <TraceNode title="Generation" active={hasGeneration || Boolean(response.answer)} />
        <TraceConnector />
        <TraceNode title="Final Answer" active={Boolean(response.answer)} />

        {hasSteps ? (
          <>
            <Separator />
            <div className="space-y-3">
              <p className="text-sm font-semibold text-slate-900">Execution steps</p>
              {steps.map(renderStep)}
            </div>
          </>
        ) : null}

        <Separator />
        <div className="flex flex-wrap gap-2 text-xs text-slate-500">
          {typeof metadata.step_count === "number" ? (
            <Badge variant="outline">Steps: {metadata.step_count}</Badge>
          ) : null}
          {typeof metadata.citation_count === "number" ? (
            <Badge variant="outline">Citations: {metadata.citation_count}</Badge>
          ) : null}
          {tool_used ? (
            <Badge variant="outline">Tools: {formatToolLabel(tool_used)}</Badge>
          ) : null}
          {metadata.partial_success === true ? (
            <Badge variant="warning">Partial success</Badge>
          ) : null}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function TraceNode({ title, active }: { title: string; active: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`h-2.5 w-2.5 rounded-full ${active ? "bg-emerald-500" : "bg-slate-300"}`}
      />
      <span className={`text-sm ${active ? "font-medium text-slate-900" : "text-slate-400"}`}>
        {title}
      </span>
    </div>
  );
}

function TraceConnector() {
  return <div className="ml-1 h-4 border-l border-dashed border-slate-300" />;
}
