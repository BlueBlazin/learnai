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
});
