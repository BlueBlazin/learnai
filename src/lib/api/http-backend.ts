import type { Backend, StartRunArgs } from "./backend";
import type { Framework, InstallEvent, RunEvent } from "./types";

const DEFAULT_BASE_URL = "http://127.0.0.1:1421";
/** How long the first load waits for the bridge before reporting it unreachable. */
const CONNECT_TIMEOUT_MS = 30_000;
/** Once the bridge has answered, a later outage is reported quickly. */
const RECONNECT_TIMEOUT_MS = 2_000;
/** The wait between probes doubles from the first value up to the second. */
const CONNECT_RETRY_MS = [250, 1_000] as const;
const POST_EMPTY = { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" } as const;

/**
 * Devbridge transport (docs/ARCHITECTURE.md section 4.1). Commands are
 * `POST /api/<command>` with a JSON body of camelCase arguments; streams are
 * server-sent events that replay from the first event and close after the
 * terminal one.
 */
export class HttpBackend implements Backend {
  readonly kind = "http" as const;
  private readonly baseUrl: string;

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  /** Resolves once the bridge has answered anything; see `connect`. */
  private connected: Promise<void> | null = null;
  private everConnected = false;

  /**
   * Waits for the bridge to accept connections. In development the UI usually
   * loads while the bridge is still compiling, so the first load polls quietly
   * (one request at a time, however many queries are waiting, and a plain GET
   * that needs no CORS preflight) and the screens stay on their loading state. After `CONNECT_TIMEOUT_MS` the calls fail and
   * the screens show their error state; "Try again" starts a new wait.
   */
  private connect(): Promise<void> {
    this.connected ??= (async () => {
      const deadline = Date.now() + (this.everConnected ? RECONNECT_TIMEOUT_MS : CONNECT_TIMEOUT_MS);
      for (let delay: number = CONNECT_RETRY_MS[0]; ; delay = Math.min(delay * 2, CONNECT_RETRY_MS[1])) {
        try {
          // Any response, even an opaque one or an error status, means the bridge is listening.
          await fetch(`${this.baseUrl}/`, { mode: "no-cors" });
          this.everConnected = true;
          return;
        } catch {
          if (Date.now() >= deadline) {
            this.connected = null;
            throw new Error(`Cannot reach the devbridge at ${this.baseUrl}. Is it running?`);
          }
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    })();
    return this.connected;
  }

  private async call<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
    await this.connect();
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/api/${command}`, { ...POST_EMPTY, body: JSON.stringify(args) });
    } catch {
      // The bridge went away after it had answered: the next call waits for it again.
      this.connected = null;
      throw new Error(`Cannot reach the devbridge at ${this.baseUrl}. Is it running?`);
    }
    const text = await response.text();
    const body: unknown = text ? JSON.parse(text) : null;
    if (!response.ok) {
      const message =
        body && typeof body === "object" && "error" in body
          ? String((body as { error: unknown }).error)
          : `${command} failed with HTTP ${response.status}`;
      throw new Error(message);
    }
    return body as T;
  }

  /**
   * Subscribes to an event stream. The bridge replays from the first event, so
   * the browser's automatic reconnect would duplicate events: on any transport
   * error the stream is closed and `onLost` reports it instead.
   */
  private subscribe<E>(
    path: string,
    onEvent: (event: E) => void,
    isTerminal: (event: E) => boolean,
    onLost: () => void,
  ): void {
    const source = new EventSource(`${this.baseUrl}${path}`);
    let finished = false;
    source.onmessage = (message) => {
      if (finished) return;
      const event = JSON.parse(message.data as string) as E;
      if (isTerminal(event)) {
        finished = true;
        source.close();
      }
      onEvent(event);
    };
    source.onerror = () => {
      if (finished) return;
      finished = true;
      source.close();
      onLost();
    };
  }

  listTracks = () => this.call<Awaited<ReturnType<Backend["listTracks"]>>>("list_tracks");
  listProblems = () => this.call<Awaited<ReturnType<Backend["listProblems"]>>>("list_problems");
  getProblem = (id: string) => this.call<Awaited<ReturnType<Backend["getProblem"]>>>("get_problem", { id });

  readSolution = (id: string, framework: Framework) =>
    this.call<Awaited<ReturnType<Backend["readSolution"]>>>("read_solution", { id, framework });
  writeSolution = async (id: string, framework: Framework, code: string) => {
    await this.call<null>("write_solution", { id, framework, code });
  };
  resetSolution = (id: string, framework: Framework) =>
    this.call<Awaited<ReturnType<Backend["resetSolution"]>>>("reset_solution", { id, framework });

  async startRun(args: StartRunArgs, onEvent: (event: RunEvent) => void): Promise<string> {
    const { runId } = await this.call<{ runId: string }>("start_run", { ...args });
    this.subscribe<RunEvent>(
      `/api/runs/${encodeURIComponent(runId)}/events`,
      onEvent,
      (event) => event.type === "end",
      () =>
        onEvent({
          type: "end",
          verdict: "harness_error",
          passed: 0,
          total: 0,
          durationMs: 0,
          metrics: [],
          submissionId: null,
          message: "Lost the connection to the devbridge.",
          detail: null,
        }),
    );
    return runId;
  }
  cancelRun = async (runId: string) => {
    await this.call<null>("cancel_run", { runId });
  };

  listSubmissions = (problemId?: string) =>
    this.call<Awaited<ReturnType<Backend["listSubmissions"]>>>(
      "list_submissions",
      problemId === undefined ? {} : { problemId },
    );
  getSubmission = (id: number) =>
    this.call<Awaited<ReturnType<Backend["getSubmission"]>>>("get_submission", { id });

  unlockEditorial = (id: string) =>
    this.call<Awaited<ReturnType<Backend["unlockEditorial"]>>>("unlock_editorial", { id });
  getReference = (id: string, framework: Framework) => this.call<string>("get_reference", { id, framework });

  envStatus = () => this.call<Awaited<ReturnType<Backend["envStatus"]>>>("env_status");
  async envInstall(framework: Framework, onEvent: (event: InstallEvent) => void): Promise<void> {
    const { jobId } = await this.call<{ jobId: string }>("env_install", { framework });
    this.subscribe<InstallEvent>(
      `/api/jobs/${encodeURIComponent(jobId)}/events`,
      onEvent,
      (event) => event.type === "done",
      () => onEvent({ type: "done", ok: false, message: "Lost the connection to the devbridge." }),
    );
  }

  getStats = () => this.call<Awaited<ReturnType<Backend["getStats"]>>>("get_stats");

  listCatalogIssues = () =>
    this.call<Awaited<ReturnType<Backend["listCatalogIssues"]>>>("list_catalog_issues");

  listPaths = () => this.call<Awaited<ReturnType<Backend["listPaths"]>>>("list_paths");
  getProblemAsset = (id: string, path: string) => this.call<string>("get_problem_asset", { id, path });
}
