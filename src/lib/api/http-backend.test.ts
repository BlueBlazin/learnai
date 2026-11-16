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

  describe("editor routes", () => {
    interface Source {
      url: string;
      readyState: number;
      onmessage?: (message: { data: string }) => void;
      onerror?: () => void;
      closed: boolean;
    }

    function stubEventSource(): Source[] {
      const sources: Source[] = [];
      class FakeEventSource {
        onmessage?: (message: { data: string }) => void;
        onerror?: () => void;
        closed = false;
        /** 0 connecting (the browser is reconnecting), 1 open, 2 closed for good. */
        readyState = 1;
        constructor(readonly url: string) {
          sources.push(this);
        }
        close() {
          this.closed = true;
          this.readyState = 2;
        }
      }
      vi.stubGlobal("EventSource", FakeEventSource);
      return sources;
    }

    const READY = {
      type: "state",
      state: {
        status: "ready",
        sessionId: "s 1",
        serverName: "ty",
        serverVersion: "0.0.104",
        rootUri: "file:///ws",
        workspacePath: "/ws",
        generation: 1,
      },
    };

    it("posts the editor commands with the contract's argument names", async () => {
      vi.useRealTimers();
      stubEventSource();
      const settings = { reviewsEnabled: true, editorIntelligence: false };
      const docs = { found: false, name: null, kind: null, signature: null, doc: null, truncated: false };
      const fetchMock = vi.fn(async (url: string, _init?: RequestInit) =>
        ok(
          url.endsWith("/set_editor_intelligence")
            ? settings
            : url.endsWith("/editor_start")
              ? { sessionId: "s 1" }
              : url.endsWith("/editor_install")
                ? { jobId: "job 3" }
                : url.endsWith("/editor_docs")
                  ? docs
                  : url.endsWith("/editor_status")
                    ? { enabled: true, installed: true, installing: false, serverVersion: null, state: null }
                    : null,
        ),
      );
      vi.stubGlobal("fetch", fetchMock);
      const backend = new HttpBackend("http://bridge");

      await expect(backend.setEditorIntelligence(false)).resolves.toEqual(settings);
      await expect(backend.editorStatus()).resolves.toMatchObject({ enabled: true, installed: true });
      await expect(backend.editorStart(() => {})).resolves.toBe("s 1");
      await expect(
        backend.editorSend("s 1", '{"jsonrpc":"2.0","method":"initialized"}'),
      ).resolves.toBeUndefined();
      await expect(backend.editorStop("s 1")).resolves.toBeUndefined();
      await expect(backend.editorStop()).resolves.toBeUndefined();
      await expect(backend.editorDocs("import torch", "torch", "exp")).resolves.toEqual(docs);
      await expect(backend.editorDocs("import torch", "torch.exp")).resolves.toEqual(docs);
      await expect(backend.editorInstall(() => {})).resolves.toBeUndefined();

      const posts = fetchMock.mock.calls
        .filter(([url]) => url.includes("/api/"))
        .map(([url, init]) => [url.replace("http://bridge/api/", ""), JSON.parse(String(init?.body))]);
      expect(posts).toEqual([
        ["set_editor_intelligence", { enabled: false }],
        ["editor_status", {}],
        ["editor_start", {}],
        ["editor_send", { sessionId: "s 1", message: '{"jsonrpc":"2.0","method":"initialized"}' }],
        ["editor_stop", { sessionId: "s 1" }],
        ["editor_stop", {}],
        ["editor_docs", { source: "import torch", expr: "torch", member: "exp" }],
        ["editor_docs", { source: "import torch", expr: "torch.exp" }],
        ["editor_install", {}],
      ]);
    });

    it("streams a session's events in order until a terminal state", async () => {
      vi.useRealTimers();
      const sources = stubEventSource();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ok({ sessionId: "s 1" })),
      );
      const backend = new HttpBackend("http://bridge");
      const events: unknown[] = [];
      await backend.editorStart((event) => events.push(event));

      const source = sources[0];
      expect(source?.url).toBe("http://bridge/api/editor/s%201/events");
      const message = { type: "message", message: '{"jsonrpc":"2.0","id":1,"result":null}' };
      const restarting = {
        type: "state",
        state: { status: "restarting", attempt: 1, maxAttempts: 4, delayMs: 500, message: "It stopped." },
      };
      const stopped = { type: "state", state: { status: "stopped", reason: "requested" } };
      for (const event of [{ type: "state", state: { status: "starting" } }, READY, message, restarting]) {
        source?.onmessage?.({ data: JSON.stringify(event) });
      }
      // `restarting` is not the end: the stream stays open for the next `ready`.
      expect(source?.closed).toBe(false);
      source?.onmessage?.({ data: JSON.stringify(stopped) });
      source?.onmessage?.({ data: JSON.stringify(message) });
      expect(events).toEqual([
        { type: "state", state: { status: "starting" } },
        READY,
        message,
        restarting,
        stopped,
      ]);
      expect(source?.closed).toBe(true);
    });

    it("ends the stream on notInstalled, the only event of a session that could not start", async () => {
      vi.useRealTimers();
      const sources = stubEventSource();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ok({ sessionId: "s 1" })),
      );
      const events: unknown[] = [];
      await new HttpBackend("http://bridge").editorStart((event) => events.push(event));
      const missing = {
        type: "state",
        state: { status: "notInstalled", message: "Install it from Settings." },
      };
      sources[0]?.onmessage?.({ data: JSON.stringify(missing) });
      expect(events).toEqual([missing]);
      expect(sources[0]?.closed).toBe(true);
    });

    it("lets the browser reconnect a session stream, and reports it lost only when it stays broken", async () => {
      vi.useRealTimers();
      const sources = stubEventSource();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ok({ sessionId: "s 1" })),
      );
      const events: { type: string; state?: { status: string; message?: string } }[] = [];
      await new HttpBackend("http://bridge").editorStart((event) => events.push(event));
      const source = sources[0];
      if (!source) throw new Error("no event source");

      // One dropped connection: the browser reconnects with Last-Event-ID and the bridge resumes.
      source.readyState = 0;
      source.onerror?.();
      source.onerror?.();
      expect(source.closed).toBe(false);
      expect(events).toEqual([]);
      source.onmessage?.({ data: JSON.stringify(READY) });
      // A message resets the count, so two more failures are still tolerated.
      source.onerror?.();
      source.onerror?.();
      expect(source.closed).toBe(false);
      source.onerror?.();
      expect(source.closed).toBe(true);
      expect(events.map((event) => event.state?.status)).toEqual(["ready", "failed"]);
      expect(events[1]?.state?.message).toBe("Lost the connection to the devbridge.");
    });

    it("reports a stream the bridge refuses as lost at once", async () => {
      vi.useRealTimers();
      const sources = stubEventSource();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ok({ sessionId: "gone" })),
      );
      const events: { state?: { status: string } }[] = [];
      await new HttpBackend("http://bridge").editorStart((event) => events.push(event as never));
      const source = sources[0];
      if (!source) throw new Error("no event source");
      source.readyState = 2;
      source.onerror?.();
      expect(events.map((event) => event.state?.status)).toEqual(["failed"]);
    });

    it("streams the editor install from the jobs endpoint until done", async () => {
      vi.useRealTimers();
      const sources = stubEventSource();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ok({ jobId: "job 3" })),
      );
      const events: unknown[] = [];
      await new HttpBackend("http://bridge").editorInstall((event) => events.push(event));
      const source = sources[0];
      expect(source?.url).toBe("http://bridge/api/jobs/job%203/events");
      const done = { type: "done", ok: true, message: null };
      source?.onmessage?.({ data: JSON.stringify({ type: "log", text: " + ty==0.0.104" }) });
      source?.onmessage?.({ data: JSON.stringify(done) });
      expect(events).toEqual([{ type: "log", text: " + ty==0.0.104" }, done]);
      expect(source?.closed).toBe(true);
    });

    it("passes 'not running' through as the error of a send", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) =>
          url.endsWith("/editor_send")
            ? new Response(JSON.stringify({ error: "the editor session is not running" }), { status: 409 })
            : ok(null),
        ),
      );
      await expect(new HttpBackend("http://bridge").editorSend("old", "{}")).rejects.toThrow(
        "the editor session is not running",
      );
    });
  });
});
