import type { RetrievalMethod } from "@/lib/types/retrieval";

export function formatRetrievalMethod(method: RetrievalMethod): string {
  switch (method) {
    case "vector":
      return "Vector";
    case "bm25":
      return "BM25";
    case "hybrid_fusion":
      return "Hybrid Fusion";
    case "multi_query":
      return "Multi-Query";
    default:
      return method;
  }
}

export function formatScore(score: number): string {
  if (Number.isInteger(score)) return String(score);
  return score.toFixed(4);
}

export function pipelineStageStatus(
  stage: { enabled: boolean; executed: boolean },
): "active" | "skipped" | "disabled" {
  if (!stage.enabled) return "disabled";
  if (stage.executed) return "active";
  return "skipped";
}
