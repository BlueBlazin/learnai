import { focusManager, QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { backend } from "./index";
import type { Framework } from "./types";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

// A desktop window regains focus without a visibility change; listen for both
// so focus-driven refetches (the solution file) work in the Tauri window.
focusManager.setEventListener((handleFocus) => {
  const onFocus = () => handleFocus(true);
  const onBlur = () => handleFocus(false);
  const onVisibility = () => handleFocus(document.visibilityState === "visible");
  window.addEventListener("focus", onFocus);
  window.addEventListener("blur", onBlur);
  document.addEventListener("visibilitychange", onVisibility);
  return () => {
    window.removeEventListener("focus", onFocus);
    window.removeEventListener("blur", onBlur);
    document.removeEventListener("visibilitychange", onVisibility);
  };
});

export const keys = {
  tracks: ["tracks"] as const,
  problems: ["problems"] as const,
  problem: (id: string) => ["problem", id] as const,
  solution: (id: string, framework: Framework) => ["solution", id, framework] as const,
  submissions: (problemId?: string) => ["submissions", problemId ?? "all"] as const,
  submission: (id: number) => ["submission", id] as const,
  reference: (id: string, framework: Framework) => ["reference", id, framework] as const,
  env: ["env"] as const,
  stats: ["stats"] as const,
  catalogIssues: ["catalogIssues"] as const,
  paths: ["paths"] as const,
  asset: (id: string, path: string) => ["asset", id, path] as const,
};

/** Everything a stored submission or a status change can affect. */
export function invalidateProgress(client: QueryClient, problemId: string): Promise<unknown> {
  return Promise.all([
    client.invalidateQueries({ queryKey: keys.tracks }),
    client.invalidateQueries({ queryKey: keys.problems }),
    client.invalidateQueries({ queryKey: keys.problem(problemId) }),
    client.invalidateQueries({ queryKey: ["submissions"] }),
    client.invalidateQueries({ queryKey: keys.stats }),
    client.invalidateQueries({ queryKey: keys.paths }),
  ]);
}

export function useTracks() {
  return useQuery({ queryKey: keys.tracks, queryFn: () => backend.listTracks() });
}

export function useProblems() {
  return useQuery({ queryKey: keys.problems, queryFn: () => backend.listProblems() });
}

export function useProblem(id: string) {
  return useQuery({ queryKey: keys.problem(id), queryFn: () => backend.getProblem(id) });
}

export function useSolution(id: string, framework: Framework | null) {
  return useQuery({
    queryKey: keys.solution(id, framework ?? "python"),
    queryFn: () => backend.readSolution(id, framework as Framework),
    enabled: framework !== null,
    // The file may be edited in another editor: re-read whenever the window regains focus.
    refetchOnWindowFocus: "always",
    staleTime: 0,
  });
}

export function useSubmissions(problemId?: string) {
  return useQuery({
    queryKey: keys.submissions(problemId),
    queryFn: () => backend.listSubmissions(problemId),
  });
}

export function useSubmission(id: number | null) {
  return useQuery({
    queryKey: keys.submission(id ?? -1),
    queryFn: () => backend.getSubmission(id as number),
    enabled: id !== null,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useReference(id: string, framework: Framework | null, enabled: boolean) {
  return useQuery({
    queryKey: keys.reference(id, framework ?? "python"),
    queryFn: () => backend.getReference(id, framework as Framework),
    enabled: enabled && framework !== null,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useEnvStatus() {
  return useQuery({ queryKey: keys.env, queryFn: () => backend.envStatus() });
}

export function useStats() {
  return useQuery({ queryKey: keys.stats, queryFn: () => backend.getStats() });
}

export function useCatalogIssues() {
  return useQuery({ queryKey: keys.catalogIssues, queryFn: () => backend.listCatalogIssues() });
}

export function usePaths() {
  return useQuery({ queryKey: keys.paths, queryFn: () => backend.listPaths() });
}

/** A statement figure as a `data:` URL. `problemId` null: the prose has no problem to resolve against. */
export function useProblemAsset(problemId: string | null, path: string) {
  return useQuery({
    queryKey: keys.asset(problemId ?? "", path),
    queryFn: () => backend.getProblemAsset(problemId as string, path),
    enabled: problemId !== null,
    // Long enough that tab switches never refetch; short enough that an author sees an edit.
    staleTime: 5 * 60_000,
  });
}

export function useUnlockEditorial(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => backend.unlockEditorial(id),
    onSuccess: (detail) => client.setQueryData(keys.problem(id), detail),
  });
}
