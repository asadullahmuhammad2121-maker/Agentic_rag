import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatToolLabel(toolUsed: string | null | undefined): string {
  if (!toolUsed) return "Unknown";
  if (toolUsed.includes("+")) return "RAG + Tavily";
  if (toolUsed === "rag_retrieval") return "RAG";
  if (toolUsed === "tavily_web_search") return "Tavily Web Search";
  return toolUsed;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function formatFileType(fileType: string): string {
  switch (fileType) {
    case "pdf":
      return "PDF";
    case "docx":
      return "Word";
    case "txt":
      return "Text";
    case "markdown":
      return "Markdown";
    case "csv":
      return "CSV";
    case "json":
      return "JSON";
    default:
      return fileType.toUpperCase();
  }
}

export function formatActionType(type: string): string {
  switch (type) {
    case "call_tool":
      return "Tool Call";
    case "call_tools":
      return "Multi-Tool Call";
    case "execute_plan":
      return "Execute Plan";
    case "finish":
      return "Finish";
    default:
      return type;
  }
}
