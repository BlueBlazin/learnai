import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpBackend } from "./http-backend";

const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

describe("HttpBackend first load", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("waits quietly for the bridge with one probe, however many calls are pending", async () => {
    let listening = false;
    const fetchMock = vi.fn(async (url: string) => {
      if (!listening) throw new TypeError("Failed to fetch");
      return ok(url.endsWith("/list_tracks") ? ["tracks"] : []);
    });
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    const tracks = backend.listTracks();
    const problems = backend.listProblems();
    await vi.advanceTimersByTimeAsync(1_200);
    // Three probes (at 0, 250, and 750 ms), not three per pending call.
    expect(fetchMock).toHaveBeenCalledTimes(3);

    listening = true;
    await vi.advanceTimersByTimeAsync(600);
    await expect(tracks).resolves.toEqual(["tracks"]);
    await expect(problems).resolves.toEqual([]);
  });

  it("gives up after the timeout, and a retry starts a new wait", async () => {
    const fetchMock = vi.fn(async (): Promise<Response> => {
      throw new TypeError("Failed to fetch");
    });
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    const first = backend.listTracks();
    const failure = expect(first).rejects.toThrow("Cannot reach the devbridge at http://bridge");
    await vi.advanceTimersByTimeAsync(31_000);
    await failure;

    fetchMock.mockImplementation(async () => ok([]));
    await expect(backend.listTracks()).resolves.toEqual([]);
  });

  it("posts list_paths and get_problem_asset with the contract's argument names", async () => {
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) =>
      ok(url.endsWith("/get_problem_asset") ? "data:image/svg+xml;base64,PHN2Zy8+" : []),
    );
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    await expect(backend.listPaths()).resolves.toEqual([]);
    await expect(backend.getProblemAsset("rope", "pair-rotation.svg")).resolves.toBe(
      "data:image/svg+xml;base64,PHN2Zy8+",
    );
    const posts = fetchMock.mock.calls
      .filter(([url]) => url.includes("/api/"))
      .map(([url, init]) => [url, JSON.parse(String(init?.body))]);
    expect(posts).toEqual([
      ["http://bridge/api/list_paths", {}],
      ["http://bridge/api/get_problem_asset", { id: "rope", path: "pair-rotation.svg" }],
    ]);
  });

  it("posts the model asset commands with the contract's argument names", async () => {
    const listing = { modelsDir: "/m", assets: [] };
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) =>
      ok(url.endsWith("/cancel_asset_fetch") ? null : listing),
    );
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    await expect(backend.listAssets()).resolves.toEqual(listing);
    await expect(backend.removeAsset("granite-4.0-350m")).resolves.toEqual(listing);
    await expect(backend.cancelAssetFetch("job-1")).resolves.toBeUndefined();
    const posts = fetchMock.mock.calls
      .filter(([url]) => url.includes("/api/"))
      .map(([url, init]) => [url, JSON.parse(String(init?.body))]);
    expect(posts).toEqual([
      ["http://bridge/api/list_assets", {}],
      ["http://bridge/api/remove_asset", { name: "granite-4.0-350m" }],
      ["http://bridge/api/cancel_asset_fetch", { jobId: "job-1" }],
    ]);
  });

  it("streams asset events from the jobs endpoint until done", async () => {
    vi.useRealTimers();
    const sources: { url: string; onmessage?: (m: { data: string }) => void; closed: boolean }[] = [];
    class FakeEventSource {
      onmessage?: (message: { data: string }) => void;
      onerror?: () => void;
      closed = false;
      constructor(readonly url: string) {
        sources.push(this);
      }
      close() {
        this.closed = true;
      }
    }
    vi.stubGlobal("EventSource", FakeEventSource);
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ok({ jobId: "job 7" }));
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    const events: unknown[] = [];
    await expect(backend.fetchAssets(["tiny"], (event) => events.push(event))).resolves.toBe("job 7");
    const post = fetchMock.mock.calls.find(([url]) => url.endsWith("/api/fetch_assets"));
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ names: ["tiny"] });

    const source = sources[0];
    expect(source?.url).toBe("http://bridge/api/jobs/job%207/events");
    const done = { type: "done", ok: true, canceled: false, message: null };
    source?.onmessage?.({ data: JSON.stringify({ type: "start", asset: "tiny", totalBytes: 2048 }) });
    source?.onmessage?.({ data: JSON.stringify(done) });
    // Nothing follows `done`, and the stream is closed.
    source?.onmessage?.({ data: JSON.stringify({ type: "log", text: "late" }) });
    expect(events).toEqual([{ type: "start", asset: "tiny", totalBytes: 2048 }, done]);
    expect(source?.closed).toBe(true);
  });

  it("passes the core's error message through", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.endsWith("/get_problem")
          ? new Response(JSON.stringify({ error: "Unknown problem: nope" }), { status: 404 })
          : ok([]),
      ),
    );
    await expect(new HttpBackend("http://bridge").getProblem("nope")).rejects.toThrow(
      "Unknown problem: nope",
    );
  });

  it("posts the run target commands with the contract's argument names", async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ok([]));
    vi.stubGlobal("fetch", fetchMock);
    const backend = new HttpBackend("http://bridge");

    await backend.listSshHosts();
    await backend.listTargets();
    await backend.addTarget("gpu-box");
    await backend.addTarget("gpu-box", "/srv/ws");
    await backend.removeTarget("ssh:gpu-box");
    const posts = fetchMock.mock.calls
      .filter(([url]) => url.includes("/api/"))
      .map(([url, init]) => [url.replace("http://bridge/api/", ""), JSON.parse(String(init?.body))]);
    expect(posts).toEqual([
      ["list_ssh_hosts", {}],
      ["list_targets", {}],
      ["add_target", { host: "gpu-box" }],
      ["add_target", { host: "gpu-box", remoteDir: "/srv/ws" }],
      ["remove_target", { name: "ssh:gpu-box" }],
    ]);
  });
});
