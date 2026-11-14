import { create } from "zustand";
import {
  type AssetDownload,
  type AssetFetchAction,
  type AssetFetchState,
  assetFetchReducer,
  initialAssetFetchState,
  isFetching,
} from "./asset-fetch-state";
import { backend } from "./index";
import { keys, queryClient } from "./queries";

interface AssetFetchStore extends AssetFetchState {
  dispatch: (action: AssetFetchAction) => void;
}

/**
 * One store for the whole app, not per screen: the core runs one fetch at a
 * time, and a download started in a workspace must still be visible, and
 * cancelable, from Settings.
 */
const useAssetFetchStore = create<AssetFetchStore>()((set) => ({
  ...initialAssetFetchState,
  dispatch: (action) => set((state) => assetFetchReducer(state, action)),
}));

/** The fetch in flight, so a cancel requested before its id arrives is not lost. */
let live: { jobId: string | null; cancelRequested: boolean } | null = null;

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function download(names: string[]): Promise<void> {
  const { dispatch } = useAssetFetchStore.getState();
  if (live || names.length === 0) return;
  const job = { jobId: null as string | null, cancelRequested: false };
  live = job;
  dispatch({ type: "begin", names });
  try {
    const jobId = await backend.fetchAssets(names, (event) => {
      if (live !== job) return;
      dispatch({ type: "event", event });
      if (event.type === "done") {
        live = null;
        // Read the listing again, then let the finished rows show what it says.
        void queryClient
          .invalidateQueries({ queryKey: keys.assets })
          .finally(() => dispatch({ type: "settled" }));
      }
    });
    job.jobId = jobId;
    if (live === job) dispatch({ type: "started", jobId });
    if (job.cancelRequested) void backend.cancelAssetFetch(jobId).catch(() => {});
  } catch (error) {
    if (live !== job) return;
    live = null;
    dispatch({ type: "failed", message: messageOf(error) });
  }
}

function cancel(): void {
  const job = live;
  if (!job) return;
  useAssetFetchStore.getState().dispatch({ type: "canceling" });
  if (job.jobId) void backend.cancelAssetFetch(job.jobId).catch(() => {});
  else job.cancelRequested = true;
}

function dismiss(name: string): void {
  useAssetFetchStore.getState().dispatch({ type: "dismiss", name });
}

export interface AssetFetchController {
  /** Per-asset download state, by asset name. Absent: nothing is happening for it. */
  downloads: Record<string, AssetDownload>;
  /** True while a fetch is in flight; the core runs one at a time. */
  busy: boolean;
  canceling: boolean;
  /** Starts downloading the named assets. Nothing is ever downloaded without this call. */
  download: (names: string[]) => void;
  /** Stops the fetch in flight; files that were complete are kept. */
  cancel: () => void;
  /** Clears a failure message. */
  dismiss: (name: string) => void;
}

/** Model downloads: start one, follow its progress, cancel it. Shared by every screen. */
export function useAssetFetch(): AssetFetchController {
  const downloads = useAssetFetchStore((state) => state.downloads);
  const busy = useAssetFetchStore(isFetching);
  const canceling = useAssetFetchStore((state) => state.canceling);
  return { downloads, busy, canceling, download: (names) => void download(names), cancel, dismiss };
}
