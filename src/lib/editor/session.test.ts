import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EditorEvent, EditorState, EditorStatus } from "@/lib/api/types";
import { EditorSessionManager, SESSION_LINGER_MS, type SessionBackend } from "./session";

const STATUS: EditorStatus = {
  enabled: true,
  installed: true,
  installing: false,
  serverVersion: "0.0.104",
  state: null,
};

const ready = (sessionId: string, generation: number): EditorState => ({
  status: "ready",
  sessionId,
  serverName: "ty",
  serverVersion: "0.0.104",
  rootUri: "file:///ws",
  workspacePath: "/ws",
  generation,
});

/** A backend whose sessions are driven by the test: `emit(n, event)` speaks for the n-th session. */
function fakeBackend(overrides: Partial<EditorStatus> = {}) {
  const sinks: ((event: EditorEvent) => void)[] = [];
  const sent: [string, string][] = [];
  const stopped: (string | undefined)[] = [];
  let sendFails = false;
  const backend: SessionBackend = {
    editorStatus: vi.fn(async () => ({ ...STATUS, ...overrides })),
    editorStart: vi.fn(async (onEvent) => {
      sinks.push(onEvent);
      return `session-${sinks.length}`;
    }),
    editorSend: vi.fn(async (sessionId, message) => {
      if (sendFails) throw new Error("the editor session is not running");
      sent.push([sessionId, message]);
    }),
    editorStop: vi.fn(async (sessionId) => {
      stopped.push(sessionId);
    }),
  };
  return {
    backend,
    sent,
    stopped,
    failSends: (fails: boolean) => {
      sendFails = fails;
    },
    emit: (session: number, event: EditorEvent) => sinks[session - 1]?.(event),
    state: (session: number, state: EditorState) => sinks[session - 1]?.({ type: "state", state }),
    sessions: () => sinks.length,
  };
}

const flush = () => vi.advanceTimersByTimeAsync(0);

describe("EditorSessionManager", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts nothing until an editor wants it, and one session for any number of editors", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    expect(manager.getView()).toEqual({ status: "idle" });
    expect(fake.backend.editorStart).not.toHaveBeenCalled();

    manager.acquire();
    manager.acquire();
    expect(manager.getView()).toEqual({ status: "starting" });
    await flush();
    expect(fake.sessions()).toBe(1);

    fake.state(1, { status: "starting" });
    fake.state(1, ready("session-1", 1));
    const view = manager.getView();
    expect(view.status).toBe("ready");
    if (view.status !== "ready") return;
    expect(view.connection).toMatchObject({ sessionId: "session-1", generation: 1, rootUri: "file:///ws" });
  });

  it("makes a new connection for every ready generation and closes the old one", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    const changes = vi.fn();
    manager.subscribe(changes);
    manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));
    const first = manager.getView();
    if (first.status !== "ready") throw new Error("expected ready");
    const received: string[] = [];
    first.connection.transport.subscribe((message) => received.push(message));
    fake.emit(1, { type: "message", message: '{"id":1}' });
    expect(received).toEqual(['{"id":1}']);

    // The server died: the client of generation 1 must hear nothing more.
    fake.state(1, { status: "restarting", attempt: 1, maxAttempts: 4, delayMs: 500, message: "It stopped." });
    expect(manager.getView()).toMatchObject({ status: "restarting", attempt: 1, message: "It stopped." });
    expect(first.connection.transport.closed).toBe(true);
    fake.state(1, { status: "starting" });
    expect(manager.getView().status).toBe("restarting");

    fake.state(1, ready("session-1", 2));
    const second = manager.getView();
    if (second.status !== "ready") throw new Error("expected ready");
    expect(second.connection).not.toBe(first.connection);
    expect(second.connection.generation).toBe(2);
    fake.emit(1, { type: "message", message: '{"id":2}' });
    expect(received).toEqual(['{"id":1}']);

    second.connection.transport.send('{"method":"initialize"}');
    await flush();
    expect(fake.sent).toEqual([["session-1", '{"method":"initialize"}']]);
    expect(changes.mock.calls.length).toBeGreaterThanOrEqual(4);
  });

  it("survives a failed send: the message is lost and the next state decides", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));
    const view = manager.getView();
    if (view.status !== "ready") throw new Error("expected ready");
    fake.failSends(true);
    view.connection.transport.send("lost");
    await flush();
    expect(manager.getView().status).toBe("ready");
    fake.failSends(false);
    view.connection.transport.send("kept");
    await flush();
    expect(fake.sent).toEqual([["session-1", "kept"]]);
  });

  it("reports the terminal states: not installed, failed, and stopped", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    fake.state(1, { status: "notInstalled", message: "Install it from Settings." });
    expect(manager.getView()).toEqual({ status: "notInstalled", message: "Install it from Settings." });

    manager.restart();
    await flush();
    fake.state(2, ready("session-2", 1));
    fake.state(2, { status: "failed", message: "It keeps stopping." });
    expect(manager.getView()).toEqual({ status: "failed", message: "It keeps stopping." });

    manager.restart();
    await flush();
    fake.state(3, ready("session-3", 1));
    // Another window started a session: this one ends quietly and comes back on request.
    fake.state(3, { status: "stopped", reason: "replaced" });
    expect(manager.getView()).toEqual({ status: "idle" });
    manager.resume();
    await flush();
    expect(fake.sessions()).toBe(4);
  });

  it("ignores events of a session it has replaced", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));
    manager.restart();
    await flush();
    expect(fake.sessions()).toBe(2);
    // The old stream ends with `stopped: replaced` and may still carry messages.
    fake.state(1, { status: "stopped", reason: "replaced" });
    fake.state(1, ready("session-1", 9));
    expect(manager.getView()).toEqual({ status: "starting" });
    fake.state(2, ready("session-2", 1));
    expect(manager.getView()).toMatchObject({ status: "ready", connection: { sessionId: "session-2" } });
  });

  it("follows the preference: off stops listening, on starts again for the open editor", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));
    const before = manager.getView();
    manager.setEnabled(false);
    expect(manager.getView()).toEqual({ status: "off" });
    if (before.status === "ready") expect(before.connection.transport.closed).toBe(true);
    // Core ends the stream with `stopped: disabled`; nothing changes.
    fake.state(1, { status: "stopped", reason: "disabled" });
    expect(manager.getView()).toEqual({ status: "off" });
    manager.restart();
    await flush();
    expect(fake.sessions()).toBe(1);

    manager.setEnabled(true);
    expect(manager.getView()).toEqual({ status: "starting" });
    await flush();
    expect(fake.sessions()).toBe(2);
  });

  it("shows off, and starts nothing, when the preference is off at the first start", async () => {
    const fake = fakeBackend({ enabled: false });
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    expect(manager.getView()).toEqual({ status: "off" });
    expect(fake.backend.editorStart).not.toHaveBeenCalled();
  });

  it("turns a rejected start into a failed state with the message", async () => {
    const fake = fakeBackend();
    vi.mocked(fake.backend.editorStart).mockRejectedValueOnce(new Error("uv was not found"));
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    expect(manager.getView()).toEqual({ status: "failed", message: "uv was not found" });
  });

  it("restarts after an install while an editor is open, with a new session", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));
    manager.restart();
    expect(manager.getView()).toEqual({ status: "starting" });
    await flush();
    expect(fake.sessions()).toBe(2);
  });

  it("stops the server a little after the last editor closes, and not if one comes back", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    const release = manager.acquire();
    await flush();
    fake.state(1, ready("session-1", 1));

    release();
    release();
    await vi.advanceTimersByTimeAsync(SESSION_LINGER_MS - 1);
    const again = manager.acquire();
    await vi.advanceTimersByTimeAsync(SESSION_LINGER_MS * 2);
    expect(fake.stopped).toEqual([]);
    expect(manager.getView().status).toBe("ready");
    expect(fake.sessions()).toBe(1);

    again();
    await vi.advanceTimersByTimeAsync(SESSION_LINGER_MS);
    expect(fake.stopped).toEqual(["session-1"]);
    expect(manager.getView()).toEqual({ status: "idle" });
    // The stream's own `stopped: requested` arrives afterwards and changes nothing.
    fake.state(1, { status: "stopped", reason: "requested" });
    expect(manager.getView()).toEqual({ status: "idle" });
  });

  it("stops a session that was still starting when it was told to stop", async () => {
    const fake = fakeBackend();
    const manager = new EditorSessionManager(fake.backend);
    manager.acquire();
    const stopping = manager.stop();
    await flush();
    await stopping;
    expect(manager.getView()).toEqual({ status: "idle" });
    // Either the start never reached core, or the session it made was stopped by id.
    expect(fake.sessions() === 0 || fake.stopped.includes("session-1")).toBe(true);
  });
});
