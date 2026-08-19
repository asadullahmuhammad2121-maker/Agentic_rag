export interface CitationResponse {
  document_id: string;
  filename: string;
  file_type: string;
  source: string;
  page_number: number;
  section: string | null;
  chunk_index: number;
  chunk_id: string;
  score: number;
  label: string;
}

export interface AgentActionResponse {
  type: "call_tool" | "call_tools" | "execute_plan" | "finish";
  tool_name: string | null;
  tool_names: string[];
  reasoning: string | null;
}

export interface AgentObservationResponse {
  tool_name: string;
  success: boolean;
  citation_count: number;
  expression?: string | null;
  result?: number | null;
}

export interface AgentStepResponse {
  action: AgentActionResponse;
  observation: AgentObservationResponse | null;
}

export interface AgentQueryRequest {
  query: string;
  top_k?: number | null;
  document_ids?: string[] | null;
  filenames?: string[] | null;
  file_types?: string[] | null;
  sections?: string[] | null;
  filters?: Record<string, string | number> | null;
}

export interface AgentQueryResponse {
  answer: string;
  citations: CitationResponse[];
  tool_used: string | null;
  steps: AgentStepResponse[];
  metadata: Record<string, unknown>;
}

export interface ApiErrorResponse {
  error: string;
  message: string;
  details: Record<string, unknown>;
}
