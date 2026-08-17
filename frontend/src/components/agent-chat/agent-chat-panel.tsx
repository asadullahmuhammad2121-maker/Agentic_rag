"use client";

import { useMutation } from "@tanstack/react-query";
import { Loader2, Send, Sparkles } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { postAgentQuery } from "@/lib/api/agent";
import { ApiError } from "@/lib/api/client";
import type { AgentQueryResponse } from "@/lib/types/agent";
import { formatToolLabel } from "@/lib/utils";
import { AgentTrace } from "@/components/agent-chat/agent-trace";
import { CitationList, MarkdownAnswer } from "@/components/agent-chat/citations";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

function toolBadgeVariant(toolUsed: string | null): "default" | "secondary" | "success" {
  if (!toolUsed) return "secondary";
  if (toolUsed.includes("+")) return "success";
  return "default";
}

export function AgentChatPanel() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<AgentQueryResponse | null>(null);

  const mutation = useMutation({
    mutationFn: postAgentQuery,
    onSuccess: (data) => {
      setResponse(data);
      toast.success("Agent response received");
    },
    onError: (error: Error) => {
      const message =
        error instanceof ApiError ? error.message : "Failed to reach the agent API";
      toast.error(message);
    },
  });

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      toast.error("Please enter a query");
      return;
    }
    mutation.mutate({ query: trimmed });
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Sparkles className="h-4 w-4" />
            Ask the Agent
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question using internal documents, the web, or both..."
              rows={4}
              disabled={mutation.isPending}
              aria-label="Agent query"
            />
            <div className="flex justify-end">
              <Button type="submit" disabled={mutation.isPending || !query.trim()}>
                {mutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running agent...
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    Submit query
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {mutation.isPending ? (
        <Card>
          <CardContent className="flex items-center gap-3 p-8 text-slate-600">
            <Loader2 className="h-5 w-5 animate-spin" />
            Agent is planning, executing tools, and generating an answer...
          </CardContent>
        </Card>
      ) : null}

      {mutation.isError && !response ? (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-sm text-red-700">
            {mutation.error instanceof ApiError
              ? mutation.error.message
              : "Something went wrong while contacting the backend."}
          </CardContent>
        </Card>
      ) : null}

      {!mutation.isPending && !response && !mutation.isError ? (
        <Card className="border-dashed">
          <CardContent className="p-10 text-center text-slate-500">
            <Sparkles className="mx-auto mb-3 h-8 w-8 text-slate-300" />
            <p className="font-medium text-slate-700">No conversation yet</p>
            <p className="mt-1 text-sm">
              Submit a query to see the agent plan, tool usage, citations, and answer.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {response ? (
        <div className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row flex-wrap items-center gap-2 space-y-0">
              <CardTitle className="text-base">Answer</CardTitle>
              {response.tool_used ? (
                <Badge variant={toolBadgeVariant(response.tool_used)}>
                  {formatToolLabel(response.tool_used)}
                </Badge>
              ) : null}
              {typeof response.metadata.citation_count === "number" ? (
                <Badge variant="outline">{response.metadata.citation_count} citations</Badge>
              ) : null}
              {typeof response.metadata.step_count === "number" ? (
                <Badge variant="outline">{response.metadata.step_count} steps</Badge>
              ) : null}
            </CardHeader>
            <CardContent className="space-y-6">
              <MarkdownAnswer content={response.answer} />
              <CitationList citations={response.citations} />
            </CardContent>
          </Card>

          <AgentTrace response={response} />
        </div>
      ) : null}
    </div>
  );
}
