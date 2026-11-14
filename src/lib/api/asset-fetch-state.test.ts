import { describe, expect, it } from "vitest";
import {
  type AssetFetchAction,
  type AssetFetchState,
  assetFetchReducer,
  initialAssetFetchState,
  isFetching,
  isInFlight,
} from "./asset-fetch-state";
import type { AssetEvent } from "./types";

function play(actions: AssetFetchAction[], from: AssetFetchState = initialAssetFetchState): AssetFetchState {
  return actions.reduce(assetFetchReducer, from);
}

const event = (e: AssetEvent): AssetFetchAction => ({ type: "event", event: e });
const begin: AssetFetchAction = { type: "begin", names: ["granite"] };
const DONE_OK: AssetEvent = { type: "done", ok: true, canceled: false, message: null };

describe("assetFetchReducer", () => {
  it("walks preparing, downloading, finished, and settles after the listing is read again", () => {
    let state = play([begin]);
    expect(isFetching(state)).toBe(true);
    expect(state.downloads.granite).toEqual({ phase: "preparing", message: null });

    state = play([event({ type: "preparing", message: "Checking the Python environment" })], state);
    expect(state.downloads.granite).toEqual({
      phase: "preparing",
      message: "Checking the Python environment",
    });

    state = play([{ type: "started", jobId: "job-1" }], state);
    expect(state.jobId).toBe("job-1");

    state = play([event({ type: "start", asset: "granite", totalBytes: 1000 })], state);
    expect(state.downloads.granite).toEqual({
      phase: "downloading",
      doneBytes: 0,
      totalBytes: 1000,
      file: null,
    });

    state = play(
      [
        event({ type: "progress", asset: "granite", file: "config.json", doneBytes: 10, totalBytes: 1000 }),
        event({ type: "log", text: "a line of output" }),
        event({
          type: "progress",
          asset: "granite",
          file: "model.safetensors",
          doneBytes: 400,
          totalBytes: 1000,
        }),
      ],
      state,
    );
    expect(state.downloads.granite).toEqual({
      phase: "downloading",
      doneBytes: 400,
      totalBytes: 1000,
      file: "model.safetensors",
    });
    expect(isInFlight(state.downloads.granite)).toBe(true);

    state = play([event({ type: "assetEnd", asset: "granite", ok: true, message: null })], state);
    expect(state.downloads.granite).toEqual({ phase: "finished" });
    expect(isFetching(state)).toBe(true);

    state = play([event(DONE_OK)], state);
    expect(isFetching(state)).toBe(false);
    expect(state.jobId).toBeNull();
    // Still shown as finished until the listing has been read again.
    expect(state.downloads.granite).toEqual({ phase: "finished" });
    expect(play([{ type: "settled" }], state).downloads).toEqual({});
  });

  it("never moves an asset's progress backward", () => {
    const state = play([
      begin,
      event({ type: "start", asset: "granite", totalBytes: 1000 }),
      event({ type: "progress", asset: "granite", file: "a", doneBytes: 600, totalBytes: 1000 }),
      event({ type: "progress", asset: "granite", file: "b", doneBytes: 200, totalBytes: 1000 }),
    ]);
    expect(state.downloads.granite).toMatchObject({ doneBytes: 600, file: "b" });
  });

  it("keeps the one-line reason of a failed asset until the next attempt or a dismissal", () => {
    const failed = play([
      begin,
      event({ type: "start", asset: "granite", totalBytes: 1000 }),
      event({ type: "assetEnd", asset: "granite", ok: false, message: "the connection was reset" }),
      event({ type: "done", ok: false, canceled: false, message: "granite: the connection was reset" }),
      { type: "settled" },
    ]);
    expect(isFetching(failed)).toBe(false);
    expect(failed.downloads.granite).toEqual({ phase: "failed", message: "the connection was reset" });

    expect(play([{ type: "dismiss", name: "granite" }], failed).downloads).toEqual({});
    // Retrying replaces the failure.
    expect(play([begin], failed).downloads.granite).toEqual({ phase: "preparing", message: null });
  });

  it("marks assets the job never reached as failed when the whole fetch fails", () => {
    const state = play([
      { type: "begin", names: ["a", "b"] },
      event({ type: "start", asset: "a", totalBytes: 10 }),
      event({ type: "assetEnd", asset: "a", ok: true, message: null }),
      event({ type: "done", ok: false, canceled: false, message: "uv was not found" }),
    ]);
    expect(state.downloads.a).toEqual({ phase: "finished" });
    expect(state.downloads.b).toEqual({ phase: "failed", message: "uv was not found" });
  });

  it("a cancel clears what was in flight and keeps what finished", () => {
    let state = play([
      { type: "begin", names: ["a", "b"] },
      event({ type: "start", asset: "a", totalBytes: 10 }),
      event({ type: "assetEnd", asset: "a", ok: true, message: null }),
      event({ type: "start", asset: "b", totalBytes: 10 }),
      { type: "canceling" },
    ]);
    expect(state.canceling).toBe(true);
    state = play([event({ type: "done", ok: false, canceled: true, message: null })], state);
    expect(state.canceling).toBe(false);
    expect(state.downloads).toEqual({ a: { phase: "finished" } });
  });

  it("reports a fetch that could not start, and ignores everything while idle", () => {
    const rejected = play([begin, { type: "failed", message: "a model download is already running" }]);
    expect(isFetching(rejected)).toBe(false);
    expect(rejected.downloads.granite).toEqual({
      phase: "failed",
      message: "a model download is already running",
    });

    const idle = play([
      event({ type: "start", asset: "granite", totalBytes: 10 }),
      { type: "started", jobId: "x" },
      { type: "canceling" },
      { type: "failed", message: "late" },
      { type: "dismiss", name: "granite" },
    ]);
    expect(idle).toEqual(initialAssetFetchState);

    // One fetch at a time: a second begin does not replace the first.
    const second = play([begin, { type: "begin", names: ["other"] }]);
    expect(second.active).toEqual(["granite"]);
    // Nothing follows `done`.
    const after = play([begin, event(DONE_OK), event({ type: "start", asset: "granite", totalBytes: 10 })]);
    expect(after.downloads).toEqual({});
  });
});
