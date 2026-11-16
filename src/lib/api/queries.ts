import { focusManager, QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { editorSession } from "@/lib/editor";
import { backend } from "./index";
import type { Framework, ReviewAction } from "./types";

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
  assets: ["assets"] as const,
  targets: ["targets"] as const,
  sshHosts: ["sshHosts"] as const,
  reviews: ["reviews"] as const,
  reviewSolution: (id: string, framework: Framework) => ["reviewSolution", id, framework] as const,
  settings: ["settings"] as const,
  editorStatus: ["editorStatus"] as const,
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
    // The first accepted submit schedules a review; an accepted review submit moves it on.
    client.invalidateQueries({ queryKey: keys.reviews }),
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

/**
 * The file the editor shows: the saved solution, or with `review` the review
 * attempt (the starter until something is written to it).
 */
export function useSolution(id: string, framework: Framework | null, review = false) {
  return useQuery({
    queryKey: review
      ? keys.reviewSolution(id, framework ?? "python")
      : keys.solution(id, framework ?? "python"),
    queryFn: () =>
      review
        ? backend.readReviewSolution(id, framework as Framework)
        : backend.readSolution(id, framework as Framework),
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

/**
 * Model assets and their state on disk. `enabled` false skips the listing for
 * screens that turn out not to need it (a problem that declares no asset).
 */
export function useAssets(enabled = true) {
  return useQuery({ queryKey: keys.assets, queryFn: () => backend.listAssets(), enabled });
}

/** Deletes one downloaded asset; the listing the core returns replaces the cached one. */
export function useRemoveAsset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => backend.removeAsset(name),
    onSuccess: (listing) => client.setQueryData(keys.assets, listing),
  });
}

/** Run targets: this computer first, then the remote machines that were added. */
export function useTargets() {
  return useQuery({ queryKey: keys.targets, queryFn: () => backend.listTargets() });
}

/**
 * Host aliases from the SSH configuration. The file is edited outside the app,
 * so it is read again whenever the window regains focus.
 */
export function useSshHosts() {
  return useQuery({
    queryKey: keys.sshHosts,
    queryFn: () => backend.listSshHosts(),
    refetchOnWindowFocus: "always",
    staleTime: 0,
  });
}

/** Adds a remote machine; the list the core returns replaces the cached one. */
export function useAddTarget() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ host, remoteDir }: { host: string; remoteDir?: string }) =>
      backend.addTarget(host, remoteDir),
    onSuccess: (targets) => client.setQueryData(keys.targets, targets),
  });
}

/** Forgets a remote machine; nothing is deleted on it. */
export function useRemoveTarget() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => backend.removeTarget(name),
    onSuccess: (targets) => client.setQueryData(keys.targets, targets),
  });
}

/** Active review schedules, soonest due first; empty while reviews are turned off. */
export function useReviews() {
  return useQuery({ queryKey: keys.reviews, queryFn: () => backend.listReviews() });
}

/** Ends a review without solving it; the list the core returns replaces the cached one. */
export function useResolveReview(problemId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (action: ReviewAction) => backend.resolveReview(problemId, action),
    onSuccess: (reviews) => {
      client.setQueryData(keys.reviews, reviews);
      void client.invalidateQueries({ queryKey: keys.stats });
    },
  });
}

export function useSettings() {
  return useQuery({ queryKey: keys.settings, queryFn: () => backend.getSettings() });
}

/** Turns reviews on or off. Everything that shows a review is read again. */
export function useSetReviewsEnabled() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (enabled: boolean) => backend.setReviewsEnabled(enabled),
    onSuccess: (settings) => {
      client.setQueryData(keys.settings, settings);
      void client.invalidateQueries({ queryKey: keys.reviews });
      void client.invalidateQueries({ queryKey: keys.stats });
    },
  });
}

/** Whether the language server behind suggestions is installed, and its version. Starts nothing. */
export function useEditorStatus() {
  return useQuery({ queryKey: keys.editorStatus, queryFn: () => backend.editorStatus() });
}

/** Turns suggestions and documentation in the editor on or off; the running session follows. */
export function useSetEditorIntelligence() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (enabled: boolean) => backend.setEditorIntelligence(enabled),
    onSuccess: (settings) => {
      client.setQueryData(keys.settings, settings);
      editorSession.setEnabled(settings.editorIntelligence);
      void client.invalidateQueries({ queryKey: keys.editorStatus });
    },
  });
}
