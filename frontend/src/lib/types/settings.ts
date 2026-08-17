export interface ToolStatus {
  name: string;
  label: string;
  enabled: boolean;
  configured: boolean;
  available: boolean;
}

export interface GeneralSettings {
  app_name: string;
  app_version: string;
  environment: string;
  log_level: string;
  request_timeout_seconds: number;
}

export interface RAGSettings {
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  chunk_min_size: number;
  chunk_max_size: number;
  semantic_similarity_threshold: number;
  retrieval_top_k: number;
  retrieval_score_threshold: number | null;
  hybrid_search_enabled: boolean;
  hybrid_top_k: number;
  vector_search_weight: number;
  keyword_search_weight: number;
  query_transformation_enabled: boolean;
  multi_query_enabled: boolean;
  multi_query_count: number;
  context_optimization_enabled: boolean;
  context_max_chunks: number;
  context_max_tokens: number;
  context_min_score: number;
  reranking_enabled: boolean;
}

export interface AgentSettings {
  agent_enabled: boolean;
  agent_max_steps: number;
  agent_routing_enabled: boolean;
  agent_planning_enabled: boolean;
  agent_runs_persistence_enabled: boolean;
  groq_model: string;
  groq_configured: boolean;
  llm_temperature: number;
  llm_max_tokens: number;
  tools: ToolStatus[];
}

export interface SearchSettings {
  vector_search_enabled: boolean;
  bm25_enabled: boolean;
  web_search_enabled: boolean;
  web_search_configured: boolean;
  tavily_max_results: number;
  tavily_search_depth: string;
  embedding_model: string;
  embedding_dimension: number;
  qdrant_collection_name: string;
}

export interface PublicSettingsResponse {
  read_only: boolean;
  general: GeneralSettings;
  rag: RAGSettings;
  agent: AgentSettings;
  search: SearchSettings;
}

export type SettingsSectionId = "general" | "rag" | "agent" | "search" | "system";
