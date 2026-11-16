import { describe, expect, it } from "vitest";
import type { EditorEvent, EditorState } from "../types";
import { FixtureBackend } from "./backend";

const SOURCE = [
  "import torch",
  "import torch.nn.functional as F",
  "",
  "def f(x):",
  "    unused = 1",
  "    y = torch.exxp(x)",
  "    z = torch.exp(x, )",
  "    return F.scaled_dot_product_attention",
  "",
].join("\n");

interface RpcMessage {
  id?: number;
  method?: string;
  result?: unknown;
  error?: { code: number };
  params?: { diagnostics?: { severity: number; message: string }[]; uri?: string };
}

/** A minimal JSON-RPC client over the backend's editor commands, enough to talk to the fake server. */
async function connect(backend: FixtureBackend) {
  const states: EditorState[] = [];
  const messages: RpcMessage[] = [];
  const waiting: { test: () => boolean; resolve: () => void }[] = [];
  const onEvent = (event: EditorEvent) => {
    if (event.type === "state") states.push(event.state);
    else messages.push(JSON.parse(event.message) as RpcMessage);
    for (const entry of [...waiting]) {
      if (!entry.test()) continue;
      waiting.splice(waiting.indexOf(entry), 1);
      entry.resolve();
    }
  };
  const until = (test: () => boolean) =>
    test() ? Promise.resolve() : new Promise<void>((resolve) => waiting.push({ test, resolve }));
  const sessionId = await backend.editorStart(onEvent);
  let nextId = 0;
  const notify = (method: string, params: unknown) =>
    backend.editorSend(sessionId, JSON.stringify({ jsonrpc: "2.0", method, params }));
  const request = async (method: string, params: unknown): Promise<RpcMessage> => {
    nextId += 1;
    const id = nextId;
    await backend.editorSend(sessionId, JSON.stringify({ jsonrpc: "2.0", id, method, params }));
    await until(() => messages.some((message) => message.id === id));
    return messages.find((message) => message.id === id) as RpcMessage;
  };
  const status = (name: EditorState["status"]) => states.filter((state) => state.status === name);
  return { sessionId, states, messages, until, notify, request, status };
}

const URI = "file:///Users/ada/Developer/learnai/workspace/stable-softmax/torch.py";
const at = (line: number, character: number) => ({
  textDocument: { uri: URI },
  position: { line, character },
});

describe("the fixture language server", () => {
  it("starts a session, answers initialize, and serves the editor features", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const client = await connect(backend);
    await client.until(() => client.status("ready").length === 1);
    expect(client.states.map((state) => state.status)).toEqual(["starting", "ready"]);
    const ready = client.status("ready")[0];
    expect(ready?.status === "ready" && [ready.sessionId, ready.generation, ready.serverName]).toEqual([
      client.sessionId,
      1,
      "ty",
    ]);

    const init = await client.request("initialize", { capabilities: {} });
    expect(init.result).toMatchObject({
      capabilities: { hoverProvider: true, completionProvider: { triggerCharacters: ["."] } },
    });
    await client.notify("initialized", {});
    await client.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: SOURCE },
    });

    // One real mistake, and a hint the client is expected to drop.
    await client.until(() => client.messages.some((m) => m.method === "textDocument/publishDiagnostics"));
    const published = client.messages.find((m) => m.method === "textDocument/publishDiagnostics");
    expect(published?.params?.uri).toBe(URI);
    expect(published?.params?.diagnostics?.map((d) => [d.severity, d.message])).toEqual([
      [1, "Module `torch` has no member `exxp`"],
      [4, "`unused` is unused"],
    ]);

    const completion = await client.request("textDocument/completion", at(5, 16));
    const items = (completion.result as { items: { label: string; documentation?: unknown }[] }).items;
    // The server lists the module first, as the real one does; ranking is the client's job.
    expect(items[0]?.label).toBe("export");
    expect(items.map((item) => item.label)).toContain("exp");
    expect(items.find((item) => item.label === "exp")?.documentation).toBeDefined();

    const hover = await client.request("textDocument/hover", at(6, 15));
    expect((hover.result as { contents: { value: string } }).contents.value).toContain("def exp(");
    const bare = await client.request("textDocument/hover", at(7, 20));
    const bareText = (bare.result as { contents: { value: string } }).contents.value;
    expect(bareText).toContain("scaled_dot_product_attention");
    expect(bareText).not.toContain("---");

    const signature = await client.request("textDocument/signatureHelp", at(6, 21));
    expect(signature.result).toMatchObject({
      activeParameter: 1,
      signatures: [{ label: "(input: Tensor, out: Tensor | None = None) -> Tensor" }],
    });

    const unknown = await client.request("workspace/symbol", { query: "x" });
    expect(unknown.error?.code).toBe(-32601);

    await backend.editorStop(client.sessionId);
    expect(client.states.at(-1)).toEqual({ status: "stopped", reason: "requested" });
    await expect(backend.editorSend(client.sessionId, "{}")).rejects.toThrow("not running");
  });

  it("answers the docs helper's lookups, by expression and by member", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const direct = await backend.editorDocs(SOURCE, "F.scaled_dot_product_attention");
    expect(direct).toMatchObject({
      found: true,
      name: "torch.nn.functional.scaled_dot_product_attention",
      kind: "function",
    });
    expect(direct.doc).toContain(".. math::");
    expect(await backend.editorDocs(SOURCE, "F", "softmax")).toMatchObject({
      found: true,
      name: "torch.nn.functional.softmax",
    });
    expect(await backend.editorDocs(SOURCE, "x")).toMatchObject({ found: false, reason: "unknown_name" });
    expect(await backend.editorDocs(SOURCE, "f(x).shape")).toMatchObject({
      found: false,
      reason: "not_a_name",
    });
  });

  it("plays a restart: the server comes back as a new generation that knows nothing", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const client = await connect(backend);
    await client.until(() => client.status("ready").length === 1);
    await client.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: "# learnai: editor_restarting\n" },
    });
    await client.until(() => client.status("ready").length === 2);
    expect(client.states.map((state) => state.status)).toEqual([
      "starting",
      "ready",
      "restarting",
      "starting",
      "ready",
    ]);
    const second = client.status("ready")[1];
    expect(second?.status === "ready" && second.generation).toBe(2);
    // Reopening the same document on the new process does not restart it again.
    await client.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: "# learnai: editor_restarting\n" },
    });
    await client.until(() => client.messages.some((m) => m.method === "textDocument/publishDiagnostics"));
    expect(client.status("restarting")).toHaveLength(1);
  });

  it("plays a failure: the session ends with failed and sends are refused", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const client = await connect(backend);
    await client.until(() => client.status("ready").length === 1);
    await client.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: "x = 1  # learnai: editor_failed\n" },
    });
    await client.until(() => client.status("failed").length === 1);
    expect(client.states.at(-1)?.status).toBe("failed");
    await expect(backend.editorSend(client.sessionId, "{}")).rejects.toThrow("not running");
  });

  it("plays 'not installed' until the server is installed from Settings", async () => {
    const backend = new FixtureBackend({ speed: 0, editorInstalled: false });
    expect(await backend.editorStatus()).toMatchObject({ installed: false, serverVersion: null });
    const first = await connect(backend);
    await first.until(() => first.states.length === 1);
    // The only event: nothing was started.
    expect(first.states.map((state) => state.status)).toEqual(["notInstalled"]);

    const log: string[] = [];
    await new Promise<void>((resolve) => {
      void backend.editorInstall((event) => {
        if (event.type === "log") log.push(event.text);
        else resolve();
      });
    });
    expect(log.at(-1)).toContain("ty==");
    expect(await backend.editorStatus()).toMatchObject({ installed: true, serverVersion: "0.0.104" });
    const second = await connect(backend);
    await second.until(() => second.status("ready").length === 1);

    await second.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: "# learnai: editor_not_installed\n" },
    });
    await second.until(() => second.messages.some((m) => m.method === "textDocument/publishDiagnostics"));
    // After an install the marker is spent: the document can stay as it is.
    expect(second.status("notInstalled")).toHaveLength(0);
    expect((await backend.editorStatus()).installed).toBe(true);
  });

  it("plays 'not installed' from a marker in the open document", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const client = await connect(backend);
    await client.until(() => client.status("ready").length === 1);
    await client.notify("textDocument/didOpen", {
      textDocument: { uri: URI, languageId: "python", version: 0, text: "# learnai: editor_not_installed\n" },
    });
    await client.until(() => client.status("notInstalled").length === 1);
    expect((await backend.editorStatus()).installed).toBe(false);
  });

  it("follows the preference and replaces a running session", async () => {
    const backend = new FixtureBackend({ speed: 0 });
    const first = await connect(backend);
    await first.until(() => first.status("ready").length === 1);
    const second = await connect(backend);
    expect(first.states.at(-1)).toEqual({ status: "stopped", reason: "replaced" });
    await second.until(() => second.status("ready").length === 1);

    expect(await backend.setEditorIntelligence(false)).toEqual({
      reviewsEnabled: true,
      editorIntelligence: false,
    });
    expect(second.states.at(-1)).toEqual({ status: "stopped", reason: "disabled" });
    await expect(backend.editorStart(() => {})).rejects.toThrow("turned off in Settings");
    await expect(backend.editorDocs(SOURCE, "torch.exp")).rejects.toThrow("turned off in Settings");
    expect((await backend.editorStatus()).enabled).toBe(false);
  });
});
