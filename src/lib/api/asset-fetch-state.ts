import type { AssetEvent } from "./types";

/**
 * Where one asset's download stands.
 *
 * preparing    requested; the environment is being checked or the file list read
 * downloading  bytes are arriving
 * finished     the judge reported it complete; kept until the asset listing has
 *              been read again, so the row never flickers back to "not downloaded"
 * failed       the one-line reason, kept until the next attempt or a dismissal
 */
export type AssetDownload =
  | { phase: "preparing"; message: string | null }
  | { phase: "downloading"; doneBytes: number; totalBytes: number; file: string | null }
  | { phase: "finished" }
  | { phase: "failed"; message: string };

export interface AssetFetchState {
  /** The assets of the fetch in flight, in request order; empty when nothing is running. */
  active: string[];
  jobId: string | null;
  canceling: boolean;
  downloads: Record<string, AssetDownload>;
}

export type AssetFetchAction =
  | { type: "begin"; names: string[] }
  | { type: "started"; jobId: string }
  | { type: "event"; event: AssetEvent }
  /** The fetch could not be started at all (transport or core error). */
  | { type: "failed"; message: string }
  | { type: "canceling" }
  /** The asset listing has been read again: finished downloads need no placeholder now. */
  | { type: "settled" }
  | { type: "dismiss"; name: string };

export const initialAssetFetchState: AssetFetchState = {
  active: [],
  jobId: null,
  canceling: false,
  downloads: {},
};

const FALLBACK_FAILURE = "The download failed.";

export function isFetching(state: AssetFetchState): boolean {
  return state.active.length > 0;
}

/** A download that is under way. */
export type ActiveDownload = Extract<AssetDownload, { phase: "preparing" | "downloading" }>;

/** True while something is in flight for the asset. */
export function isInFlight(download: AssetDownload | undefined): download is ActiveDownload {
  return download?.phase === "preparing" || download?.phase === "downloading";
}

function withDownload(state: AssetFetchState, name: string, download: AssetDownload): AssetFetchState {
  return { ...state, downloads: { ...state.downloads, [name]: download } };
}

function applyEvent(state: AssetFetchState, event: AssetEvent): AssetFetchState {
  switch (event.type) {
    case "preparing": {
      // Before any asset has started, the message describes all of them.
      const downloads = { ...state.downloads };
      for (const name of state.active) {
        if (downloads[name]?.phase === "preparing") {
          downloads[name] = { phase: "preparing", message: event.message };
        }
      }
      return { ...state, downloads };
    }
    case "start":
      return withDownload(state, event.asset, {
        phase: "downloading",
        doneBytes: 0,
        totalBytes: event.totalBytes,
        file: null,
      });
    case "progress": {
      const current = state.downloads[event.asset];
      // The fraction for an asset never goes backward, whatever order lines arrive in.
      const floor = current?.phase === "downloading" ? current.doneBytes : 0;
      return withDownload(state, event.asset, {
        phase: "downloading",
        doneBytes: Math.max(floor, event.doneBytes),
        totalBytes: event.totalBytes,
        file: event.file || null,
      });
    }
    case "assetEnd":
      return withDownload(
        state,
        event.asset,
        event.ok ? { phase: "finished" } : { phase: "failed", message: event.message ?? FALLBACK_FAILURE },
      );
    case "log":
      return state;
    case "done": {
      const downloads = { ...state.downloads };
      for (const name of state.active) {
        if (!isInFlight(downloads[name])) continue;
        // An asset the fetch never finished: canceled, or lost with the whole job.
        if (event.canceled || event.ok) delete downloads[name];
        else downloads[name] = { phase: "failed", message: event.message ?? FALLBACK_FAILURE };
      }
      return { active: [], jobId: null, canceling: false, downloads };
    }
  }
}

/** Folds the ordered asset event stream into what the Models list and the notices render. */
export function assetFetchReducer(state: AssetFetchState, action: AssetFetchAction): AssetFetchState {
  switch (action.type) {
    case "begin": {
      if (isFetching(state)) return state;
      const downloads = { ...state.downloads };
      for (const name of action.names) downloads[name] = { phase: "preparing", message: null };
      return { active: action.names, jobId: null, canceling: false, downloads };
    }
    case "started":
      return isFetching(state) ? { ...state, jobId: action.jobId } : state;
    case "event":
      // Nothing follows `done`.
      return isFetching(state) ? applyEvent(state, action.event) : state;
    case "failed": {
      if (!isFetching(state)) return state;
      const downloads = { ...state.downloads };
      for (const name of state.active) downloads[name] = { phase: "failed", message: action.message };
      return { active: [], jobId: null, canceling: false, downloads };
    }
    case "canceling":
      return isFetching(state) ? { ...state, canceling: true } : state;
    case "settled": {
      const downloads = Object.fromEntries(
        Object.entries(state.downloads).filter(([, download]) => download.phase !== "finished"),
      );
      return { ...state, downloads };
    }
    case "dismiss": {
      if (state.downloads[action.name]?.phase !== "failed") return state;
      const { [action.name]: _dismissed, ...downloads } = state.downloads;
      return { ...state, downloads };
    }
  }
}
