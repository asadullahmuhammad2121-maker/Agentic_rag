import type {
  AgentQueryResponse,
  AgentStepResponse,
  CitationResponse,
} from "@/lib/types/agent";

export type AgentRunStatus = "success" | "failure";

export interface AgentRunSummary {
  run_id: string;
  query: string;
  status: AgentRunStatus;
  started_at: string;
  completed_at: string;
  duration_ms: number | null;
  tool_used: string | null;
  step_count: number;
  citation_count: number;
  error_message: string | null;
  error_code: string | null;
}

export interface AgentRunListResponse {
  runs: AgentRunSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface AgentRunDetail extends AgentRunSummary {
  answer: string | null;
  citations: CitationResponse[];
  steps: AgentStepResponse[];
  metadata: Record<string, unknown>;
}

export interface AgentRunListParams {
  search?: string;
  status?: AgentRunStatus;
  limit?: number;
  offset?: number;
}

/** Shape compatible with AgentQueryResponse for reusing trace components. */
export type AgentRunTraceData = Pick<
  AgentQueryResponse,
  "answer" | "citations" | "tool_used" | "steps" | "metadata"
>;

export function agentRunDetailToTraceData(run: AgentRunDetail): AgentRunTraceData {
  return {
    answer: run.answer ?? "",
    citations: run.citations,
    tool_used: run.tool_used,
    steps: run.steps,
    metadata: run.metadata,
  };
}
