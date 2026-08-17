"use client";

import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { postRetrievalExplore } from "@/lib/api/retrieval";
import { ApiError } from "@/lib/api/client";
import type { RetrievalExploreRequest, RetrievalExploreResponse } from "@/lib/types/retrieval";

export function useRetrievalExplore() {
  return useMutation<RetrievalExploreResponse, Error, RetrievalExploreRequest>({
    mutationFn: postRetrievalExplore,
    onError: (error: Error) => {
      toast.error(error instanceof ApiError ? error.message : "Retrieval failed");
    },
  });
}
