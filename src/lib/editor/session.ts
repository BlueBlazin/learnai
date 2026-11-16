import type { Backend } from "@/lib/api/backend";
import type { EditorEvent, EditorState } from "@/lib/api/types";
import { SessionTransport } from "./transport";

/** A running server process waiting for (or talking to) one LSP client. */
export interface EditorConnection {
  readonly sessionId: string;
  /** 1, 2, ... per server process within the session. */
  readonly generation: number;
  readonly rootUri: string;
  readonly serverName: string;
  readonly serverVersion: string | null;
  readonly transport: SessionTransport;
}

/** What the app shows and acts on: the session reduced to the cases the UI distinguishes. */
export type EditorSessionView =
  /** Nothing runs and nothing is wrong: no editor needs the server right now. */
  | { status: "idle" }
  /** The preference is off. */
  | { status: "off" }
  | { status: "starting" }
  | { status: "ready"; connection: EditorConnection }
  | { status: "restarting"; attempt: number; maxAttempts: number; message: string }
  | { status: "notInstalled"; message: string }
  | { status: "failed"; message: string };

export type SessionBackend = Pick<Backend, "editorStatus" | "editorStart" | "editorSend" | "editorStop">;

/** How long the server keeps running after the last editor closed, so moving between problems reuses it. */
export const SESSION_LINGER_MS = 20_000;

const IDLE: EditorSessionView = { status: "idle" };
const OFF: EditorSessionView = { status: "off" };

function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/**
 * The one editor session of the app (docs/ARCHITECTURE.md section 4.4): the
 * language server is a single process, so every editor shares this manager
 * instead of starting its own.
 *
 * Editors that want suggestions `acquire` the session; the first one starts it
 * and it stops a little after the last one lets go. Each `ready` state becomes
 * a new `EditorConnection` with its own transport, because every `ready` is a
 * fresh server process: whoever holds an LSP client makes a new one per
 * connection. Events of a session that was replaced or stopped are ignored.
 */
export class EditorSessionManager {
  private view: EditorSessionView = IDLE;
  private readonly listeners = new Set<() => void>();
  private users = 0;
  /** Identifies the latest `editorStart` call; events and results of older ones are dropped. */
  private epoch = 0;
  private sessionId: string | null = null;
  private connection: EditorConnection | null = null;
  private linger: ReturnType<typeof setTimeout> | null = null;
  private startQueue: Promise<void> = Promise.resolve();

  constructor(private readonly backend: SessionBackend) {}

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getView = (): EditorSessionView => this.view;

  /**
   * Registers an editor that wants the server and returns the function that
   * lets go again. The first user starts the session unless one is already
   * under way; a session that ended (failed, not installed) gets a new try.
   */
  acquire(): () => void {
    this.users += 1;
    if (this.linger) {
      clearTimeout(this.linger);
      this.linger = null;
    }
    if (this.users === 1 && !this.live() && this.view.status !== "off") void this.start();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.users -= 1;
      if (this.users > 0 || this.linger) return;
      this.linger = setTimeout(() => {
        this.linger = null;
        if (this.users === 0) void this.stop();
      }, SESSION_LINGER_MS);
    };
  }

  /** Follows the preference. Core has already stopped the server when it was turned off. */
  setEnabled(enabled: boolean): void {
    if (!enabled) {
      this.epoch += 1;
      this.sessionId = null;
      this.dropConnection();
      this.setView(OFF);
    } else if (this.view.status === "off") {
      this.setView(IDLE);
      if (this.users > 0) void this.start();
    }
  }

  /**
   * Starts a new server process: after an install (the server does not see
   * packages that appear while it runs), or to try again after a failure.
   * Without an editor that wants it, a running server is stopped instead.
   */
  restart(): void {
    if (this.view.status === "off") return;
    if (this.users > 0) void this.start();
    else void this.stop();
  }

  /** Starts the session again if an editor wants it and nothing runs (another window took the server). */
  resume(): void {
    if (this.users > 0 && this.view.status === "idle") void this.start();
  }

  /** Stops the session and its server. Resolves once core has confirmed it. */
  async stop(): Promise<void> {
    this.epoch += 1;
    const sessionId = this.sessionId;
    this.sessionId = null;
    this.dropConnection();
    if (this.view.status !== "off") this.setView(IDLE);
    if (sessionId === null) return;
    try {
      await this.backend.editorStop(sessionId);
    } catch {
      // The session is gone either way; there is nothing to tell the user.
    }
  }

  private live(): boolean {
    const { status } = this.view;
    return status === "starting" || status === "ready" || status === "restarting";
  }

  /**
   * Starts a session, one start at a time: a second call waits for the first
   * `editor_start` to resolve, so core always sees them in the order they were
   * made and the latest one is the session that survives.
   */
  private start(): Promise<void> {
    this.epoch += 1;
    const epoch = this.epoch;
    this.sessionId = null;
    this.dropConnection();
    this.setView({ status: "starting" });
    this.startQueue = this.startQueue.then(() => this.open(epoch));
    return this.startQueue;
  }

  private async open(epoch: number): Promise<void> {
    if (epoch !== this.epoch) return;
    try {
      // `editor_start` rejects while the preference is off; ask first so that
      // case is a state, not an error message to recognize.
      const status = await this.backend.editorStatus();
      if (epoch !== this.epoch) return;
      if (!status.enabled) {
        this.setView(OFF);
        return;
      }
      const sessionId = await this.backend.editorStart((event) => this.receive(epoch, event));
      if (epoch === this.epoch) {
        this.sessionId ??= sessionId;
        return;
      }
      // Stopped or superseded while starting: a session nobody listens to must not keep a server alive.
      await this.backend.editorStop(sessionId).catch(() => {});
    } catch (cause) {
      if (epoch !== this.epoch) return;
      this.dropConnection();
      this.setView({ status: "failed", message: messageOf(cause) });
    }
  }

  private receive(epoch: number, event: EditorEvent): void {
    if (epoch !== this.epoch) return;
    if (event.type === "message") {
      this.connection?.transport.deliver(event.message);
      return;
    }
    this.adopt(event.state);
  }

  private adopt(state: EditorState): void {
    switch (state.status) {
      case "starting":
        this.dropConnection();
        // Between `restarting` and the next `ready`, keep saying that it is restarting.
        if (this.view.status !== "restarting") this.setView({ status: "starting" });
        return;
      case "ready": {
        this.dropConnection();
        const { sessionId } = state;
        this.sessionId = sessionId;
        this.connection = {
          sessionId,
          generation: state.generation,
          rootUri: state.rootUri,
          serverName: state.serverName,
          serverVersion: state.serverVersion,
          transport: new SessionTransport((message) => this.backend.editorSend(sessionId, message)),
        };
        this.setView({ status: "ready", connection: this.connection });
        return;
      }
      case "restarting":
        this.dropConnection();
        this.setView({
          status: "restarting",
          attempt: state.attempt,
          maxAttempts: state.maxAttempts,
          message: state.message,
        });
        return;
      case "notInstalled":
      case "failed":
        this.sessionId = null;
        this.dropConnection();
        this.setView({ status: state.status, message: state.message });
        return;
      case "stopped":
        this.sessionId = null;
        this.dropConnection();
        this.setView(state.reason === "disabled" ? OFF : IDLE);
        return;
    }
  }

  private dropConnection(): void {
    this.connection?.transport.close();
    this.connection = null;
  }

  private setView(view: EditorSessionView): void {
    if (this.view === view) return;
    this.view = view;
    for (const listener of [...this.listeners]) listener();
  }
}
