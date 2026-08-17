/** Matches backend ``RetrievalMethod``. */
export type RetrievalMethod = "vector" | "bm25" | "hybrid_fusion" | "multi_query";

export interface RetrievedChunkResult {
  chunk_id: string;
  text: string;
  document_id: string;
  filename: string;
  file_type: string;
  source: string;
  page_number: number;
  section: string | null;
  chunk_index: number;
  chunking_strategy: string;
  score: number;
  retrieval_method: RetrievalMethod;
}

export interface PipelineStage {
  id: string;
  label: string;
  enabled: boolean;
  executed: boolean;
  result_count: number | null;
  details: Record<string, unknown>;
}

export interface RetrievalConfiguration {
  query_transformation_enabled: boolean;
  multi_query_enabled: boolean;
  hybrid_search_enabled: boolean;
  context_optimization_enabled: boolean;
  reranking_enabled: boolean;
}

export interface RetrievalExploreRequest {
  query: string;
  top_k?: number;
  document_ids?: string[];
  filenames?: string[];
  file_types?: string[];
  sections?: string[];
}

export interface RetrievalExploreResponse {
  query: string;
  retrieval_query: string;
  generated_queries: string[] | null;
  configuration: RetrievalConfiguration;
  pipeline: PipelineStage[];
  vector_results: RetrievedChunkResult[];
  bm25_results: RetrievedChunkResult[];
  fused_results: RetrievedChunkResult[] | null;
  results: RetrievedChunkResult[];
  metadata: Record<string, unknown>;
}

export type RetrievalResultView = "final" | "vector" | "bm25" | "fused";

export type RetrievalSortField = "score" | "filename" | "page_number";
export type RetrievalSortDirection = "asc" | "desc";

export type RetrievalMethodFilter = "all" | RetrievalMethod;
