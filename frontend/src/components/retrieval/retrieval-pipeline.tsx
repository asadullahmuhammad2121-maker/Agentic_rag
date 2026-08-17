"use client";

import { ArrowRight, CheckCircle2, Circle, MinusCircle } from "lucide-react";
import type { PipelineStage, RetrievalConfiguration } from "@/lib/types/retrieval";
import { pipelineStageStatus } from "@/lib/retrieval/utils";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface RetrievalPipelineProps {
  pipeline: PipelineStage[];
  configuration: RetrievalConfiguration;
}

const VISUAL_STAGE_IDS = [
  "query",
  "query_transformation",
  "multi_query",
  "vector_search",
  "bm25",
  "hybrid_fusion",
  "reranking",
  "context_optimization",
  "multi_query_combine",
  "final_results",
] as const;

function stageIcon(status: ReturnType<typeof pipelineStageStatus>) {
  if (status === "active") {
    return <CheckCircle2 className="h-4 w-4 text-emerald-600" aria-hidden />;
  }
  if (status === "skipped") {
    return <MinusCircle className="h-4 w-4 text-amber-500" aria-hidden />;
  }
  return <Circle className="h-4 w-4 text-slate-300" aria-hidden />;
}

function shouldShowStage(stage: PipelineStage, configuration: RetrievalConfiguration): boolean {
  switch (stage.id) {
    case "query_transformation":
      return configuration.query_transformation_enabled || stage.executed;
    case "multi_query":
    case "multi_query_combine":
      return configuration.multi_query_enabled || stage.executed;
    case "bm25":
    case "hybrid_fusion":
      return configuration.hybrid_search_enabled || stage.executed;
    case "reranking":
      return configuration.reranking_enabled;
    case "context_optimization":
      return configuration.context_optimization_enabled || stage.executed;
    default:
      return true;
  }
}

export function RetrievalPipeline({ pipeline, configuration }: RetrievalPipelineProps) {
  const stageById = new Map(pipeline.map((stage) => [stage.id, stage]));
  const visibleStages = VISUAL_STAGE_IDS.map((id) => stageById.get(id)).filter(
    (stage): stage is PipelineStage =>
      stage !== undefined && shouldShowStage(stage, configuration),
  );

  if (visibleStages.length === 0) {
    return null;
  }

  return (
    <div className="overflow-x-auto">
      <div
        className="flex min-w-max items-stretch gap-2 pb-2"
        role="list"
        aria-label="Retrieval pipeline stages"
      >
        {visibleStages.map((stage, index) => {
          const status = pipelineStageStatus(stage);
          return (
            <div key={stage.id} className="flex items-center gap-2" role="listitem">
              <div
                className={cn(
                  "flex min-w-[9rem] flex-col rounded-lg border px-3 py-3",
                  status === "active" && "border-emerald-200 bg-emerald-50",
                  status === "skipped" && "border-amber-200 bg-amber-50",
                  status === "disabled" && "border-slate-200 bg-slate-50 opacity-70",
                )}
              >
                <div className="flex items-center gap-2">
                  {stageIcon(status)}
                  <span className="text-sm font-medium text-slate-900">{stage.label}</span>
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {!stage.enabled ? (
                    <Badge variant="secondary" className="text-[10px]">
                      Off
                    </Badge>
                  ) : null}
                  {stage.result_count !== null ? (
                    <Badge variant="outline" className="text-[10px]">
                      {stage.result_count} hits
                    </Badge>
                  ) : null}
                </div>
                {stage.id === "query_transformation" &&
                typeof stage.details.retrieval_query === "string" &&
                stage.details.was_transformed === true ? (
                  <p className="mt-2 line-clamp-2 text-xs text-slate-500">
                    → {stage.details.retrieval_query as string}
                  </p>
                ) : null}
                {stage.id === "multi_query" && Array.isArray(stage.details.queries) ? (
                  <p className="mt-2 text-xs text-slate-500">
                    {(stage.details.queries as string[]).length} queries generated
                  </p>
                ) : null}
              </div>
              {index < visibleStages.length - 1 ? (
                <ArrowRight className="h-4 w-4 shrink-0 text-slate-300" aria-hidden />
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
