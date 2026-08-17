"use client";

import { useState } from "react";
import { AlertCircle, Info, Lock } from "lucide-react";
import { getApiBaseUrl } from "@/lib/api/client";
import { usePublicSettings, useSystemStatus } from "@/lib/hooks/use-settings";
import type { PublicSettingsResponse, SettingsSectionId } from "@/lib/types/settings";
import { formatBoolean, formatOptionalNumber, formatStatusLabel } from "@/lib/settings/utils";
import {
  SettingsField,
  SettingsSectionCard,
  StatusBadge,
} from "@/components/settings/settings-field";
import {
  SettingsSectionNav,
  SettingsSectionTabs,
} from "@/components/settings/settings-section-nav";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

function GeneralSection({
  settings,
  backendReachable,
}: {
  settings: PublicSettingsResponse;
  backendReachable: boolean;
}) {
  return (
    <SettingsSectionCard
      title="General"
      description="Application identity and backend connectivity. Values are read-only and configured on the server."
    >
      <SettingsField label="API base URL" value={getApiBaseUrl()} description="Frontend proxy path to the backend" />
      <SettingsField label="Backend connection" value={backendReachable ? "Connected" : "Unreachable"} />
      <SettingsField label="Application" value={settings.general.app_name} />
      <SettingsField label="Version" value={settings.general.app_version} />
      <SettingsField label="Environment" value={settings.general.environment} />
      <SettingsField label="Log level" value={settings.general.log_level} />
      <SettingsField
        label="Request timeout"
        value={`${settings.general.request_timeout_seconds}s`}
      />
    </SettingsSectionCard>
  );
}

function RAGSection({ settings }: { settings: PublicSettingsResponse }) {
  const rag = settings.rag;
  return (
    <SettingsSectionCard
      title="RAG"
      description="Retrieval and chunking configuration from backend settings."
    >
      <SettingsField label="Chunking strategy" value={rag.chunking_strategy} />
      <SettingsField label="Chunk size" value={rag.chunk_size} />
      <SettingsField label="Chunk overlap" value={rag.chunk_overlap} />
      <SettingsField label="Chunk min size" value={rag.chunk_min_size} />
      <SettingsField label="Chunk max size" value={rag.chunk_max_size} />
      <SettingsField label="Semantic similarity threshold" value={rag.semantic_similarity_threshold} />
      <SettingsField label="Retrieval top-K" value={rag.retrieval_top_k} />
      <SettingsField
        label="Retrieval score threshold"
        value={formatOptionalNumber(rag.retrieval_score_threshold)}
      />
      <SettingsField label="Hybrid search" value={formatBoolean(rag.hybrid_search_enabled)} />
      <SettingsField label="Hybrid top-K" value={rag.hybrid_top_k} />
      <SettingsField label="Vector search weight" value={rag.vector_search_weight} />
      <SettingsField label="BM25 weight" value={rag.keyword_search_weight} />
      <SettingsField
        label="Query transformation"
        value={formatBoolean(rag.query_transformation_enabled)}
      />
      <SettingsField label="Multi-query retrieval" value={formatBoolean(rag.multi_query_enabled)} />
      <SettingsField label="Multi-query count" value={rag.multi_query_count} />
      <SettingsField
        label="Context optimization"
        value={formatBoolean(rag.context_optimization_enabled)}
      />
      <SettingsField label="Context max chunks" value={rag.context_max_chunks} />
      <SettingsField label="Context max tokens" value={rag.context_max_tokens} />
      <SettingsField label="Context min score" value={rag.context_min_score} />
      <SettingsField
        label="Reranking"
        value={formatBoolean(rag.reranking_enabled)}
        description="Not configured in the current backend"
      />
    </SettingsSectionCard>
  );
}

function AgentSection({ settings }: { settings: PublicSettingsResponse }) {
  const agent = settings.agent;
  return (
    <div className="space-y-6">
      <SettingsSectionCard
        title="Agent"
        description="Agent orchestration limits and tool availability."
      >
        <SettingsField label="Agent service" value={formatBoolean(agent.agent_enabled)} />
        <SettingsField label="Max steps" value={agent.agent_max_steps} />
        <SettingsField label="LLM routing" value={formatBoolean(agent.agent_routing_enabled)} />
        <SettingsField label="Query planning" value={formatBoolean(agent.agent_planning_enabled)} />
        <SettingsField
          label="Run persistence"
          value={formatBoolean(agent.agent_runs_persistence_enabled)}
        />
        <SettingsField label="Groq model" value={agent.groq_model} />
        <SettingsField
          label="Groq configured"
          value={formatBoolean(agent.groq_configured)}
          description="Indicates whether an API key is present, not the key itself"
        />
        <SettingsField label="LLM temperature" value={agent.llm_temperature} />
        <SettingsField label="LLM max tokens" value={agent.llm_max_tokens} />
      </SettingsSectionCard>

      <SettingsSectionCard title="Available tools" description="Registered agent tools at runtime.">
        {agent.tools.map((tool) => (
          <div
            key={tool.name}
            className="flex flex-col gap-2 border-b border-slate-100 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between"
          >
            <div>
              <p className="text-sm font-medium text-slate-900">{tool.label}</p>
              <p className="text-xs text-slate-500">{tool.name}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusBadge active={tool.available} activeLabel="Available" inactiveLabel="Unavailable" />
              <StatusBadge active={tool.configured} activeLabel="Configured" inactiveLabel="Not configured" />
              <Badge variant="outline">{formatBoolean(tool.enabled)}</Badge>
            </div>
          </div>
        ))}
      </SettingsSectionCard>
    </div>
  );
}

function SearchSection({ settings }: { settings: PublicSettingsResponse }) {
  const search = settings.search;
  return (
    <SettingsSectionCard
      title="Search"
      description="Vector, keyword, and web search configuration."
    >
      <SettingsField label="Vector search" value={formatBoolean(search.vector_search_enabled)} />
      <SettingsField label="BM25 keyword search" value={formatBoolean(search.bm25_enabled)} />
      <SettingsField label="Web search (Tavily)" value={formatBoolean(search.web_search_enabled)} />
      <SettingsField
        label="Web search configured"
        value={formatBoolean(search.web_search_configured)}
        description="Requires Tavily to be enabled and an API key on the server"
      />
      <SettingsField label="Tavily max results" value={search.tavily_max_results} />
      <SettingsField label="Tavily search depth" value={search.tavily_search_depth} />
      <SettingsField label="Embedding model" value={search.embedding_model} />
      <SettingsField label="Embedding dimension" value={search.embedding_dimension} />
      <SettingsField label="Qdrant collection" value={search.qdrant_collection_name} />
    </SettingsSectionCard>
  );
}

function SystemSection({
  backendReachable,
  systemStatus,
}: {
  backendReachable: boolean;
  systemStatus: ReturnType<typeof useSystemStatus>["data"];
}) {
  const health = systemStatus?.health;
  const components = health?.components ?? [];

  return (
    <div className="space-y-6">
      <SettingsSectionCard title="System health" description="Live dependency and probe status.">
        <SettingsField label="API reachable" value={backendReachable ? "Yes" : "No"} />
        <SettingsField
          label="Health status"
          value={health ? formatStatusLabel(health.status) : "Unknown"}
        />
        <SettingsField
          label="Readiness"
          value={
            systemStatus?.readyReachable
              ? formatStatusLabel(systemStatus.ready?.status ?? "unknown")
              : "Unreachable"
          }
        />
        <SettingsField
          label="Liveness"
          value={
            systemStatus?.liveReachable
              ? formatStatusLabel(systemStatus.live?.status ?? "unknown")
              : "Unreachable"
          }
        />
      </SettingsSectionCard>

      <SettingsSectionCard title="Dependencies" description="Component checks from /health.">
        {components.length === 0 ? (
          <p className="py-4 text-sm text-slate-500">No component data available.</p>
        ) : (
          components.map((component) => (
            <div
              key={component.name}
              className="flex flex-col gap-1 border-b border-slate-100 py-3 last:border-b-0 sm:flex-row sm:items-center sm:justify-between"
            >
              <div>
                <p className="text-sm font-medium text-slate-900">
                  {component.name === "qdrant"
                    ? "Qdrant"
                    : component.name === "keyword_index"
                      ? "BM25 index"
                      : component.name}
                </p>
                {component.detail ? (
                  <p className="text-xs text-red-600">{component.detail}</p>
                ) : null}
              </div>
              <Badge
                variant={
                  component.status === "ok"
                    ? "success"
                    : component.status === "degraded"
                      ? "warning"
                      : "destructive"
                }
              >
                {formatStatusLabel(component.status)}
              </Badge>
            </div>
          ))
        )}
      </SettingsSectionCard>
    </div>
  );
}

export function SettingsPageContent() {
  const [section, setSection] = useState<SettingsSectionId>("general");
  const settingsQuery = usePublicSettings();
  const systemQuery = useSystemStatus();

  const backendReachable = !settingsQuery.isError && Boolean(settingsQuery.data);
  const isLoading = settingsQuery.isLoading || systemQuery.isLoading;
  const isError = settingsQuery.isError;

  if (isLoading) {
    return (
      <div className="mx-auto max-w-6xl space-y-4">
        <Skeleton className="h-10 w-full lg:hidden" />
        <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
          <Skeleton className="hidden h-64 lg:block" />
          <Skeleton className="h-96 w-full" />
        </div>
      </div>
    );
  }

  if (isError || !settingsQuery.data) {
    return (
      <Card className="mx-auto max-w-3xl border-red-200 bg-red-50">
        <CardContent className="flex gap-3 p-6 text-sm text-red-700">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-medium">Could not load settings</p>
            <p className="mt-1">
              Ensure the backend is running and reachable at{" "}
              <code className="rounded bg-red-100 px-1">{getApiBaseUrl()}</code>.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const settings = settingsQuery.data;

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div
        role="note"
        className="flex gap-3 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600"
      >
        <Lock className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden />
        <p>
          All settings are <strong>read-only</strong>. They reflect backend environment
          configuration and cannot be changed from this UI. API keys and secrets are never
          displayed.
        </p>
      </div>

      <SettingsSectionTabs active={section} onChange={setSection} />

      <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
        <div className="hidden lg:block">
          <SettingsSectionNav active={section} onChange={setSection} />
        </div>

        <div className="min-w-0 space-y-6">
          {section === "general" ? (
            <GeneralSection settings={settings} backendReachable={backendReachable} />
          ) : null}
          {section === "rag" ? <RAGSection settings={settings} /> : null}
          {section === "agent" ? <AgentSection settings={settings} /> : null}
          {section === "search" ? <SearchSection settings={settings} /> : null}
          {section === "system" ? (
            <SystemSection backendReachable={backendReachable} systemStatus={systemQuery.data} />
          ) : null}

          <div className="flex items-start gap-2 rounded-lg border border-dashed border-slate-200 p-4 text-xs text-slate-500">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <p>
              To change these values, update backend environment variables and restart the API
              service. Frontend{" "}
              <code className="rounded bg-slate-100 px-1">NEXT_PUBLIC_API_URL</code> only controls
              where this UI sends requests.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
