import type { AgentRunStatus } from "@/lib/types/agent-runs";
import { formatToolLabel } from "@/lib/utils";

export function formatRunStatus(status: AgentRunStatus): string {
  return status === "success" ? "Success" : "Failed";
}

export function formatDuration(durationMs: number | null): string {
  if (durationMs === null) return "—";
  if (durationMs < 1000) return `${durationMs} ms`;
  return `${(durationMs / 1000).toFixed(1)} s`;
}

export function formatRunTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function formatToolsUsed(toolUsed: string | null): string {
  if (!toolUsed) return "—";
  return formatToolLabel(toolUsed);
}
