import type { EditorDocs, EditorEvent, EditorState, EditorStatus, InstallEvent } from "../types";

/**
 * A stand-in for the language server and the docs helper (docs/ARCHITECTURE.md
 * section 4.4), small enough to read: it knows a handful of `torch` members
 * and answers `initialize`, completion, hover, signature help, and diagnostics
 * for them, so the fixture preview and the tests exercise the real client.
 *
 * A `# learnai: <scenario>` comment in the open document plays a session state,
 * the same way run scenarios are chosen: `editor_restarting` (the server dies
 * and comes back as a new generation), `editor_failed` (it dies for good), and
 * `editor_not_installed` (the server is missing until it is installed from
 * Settings). Each plays once per session.
 */

const ROOT_URI = "file:///Users/ada/Developer/learnai/workspace";
const SERVER_VERSION = "0.0.104";

interface Member {
  kind: number;
  /** What the server shows as the item's detail and on top of a hover. */
  signature: string;
  /** Parameters as they appear in signature help. */
  parameters?: string[];
  result?: string;
  /** Markdown, the way the server converts docstrings. Null: the server has none (the docs helper might). */
  doc: string | null;
}

const FUNCTION = 3;
const CLASS = 7;
const MODULE = 9;

const TORCH: Record<string, Member> = {
  exp: {
    kind: FUNCTION,
    signature: "def exp(input: Tensor, *, out: Tensor | None = None) -> Tensor",
    parameters: ["input: Tensor", "out: Tensor | None = None"],
    result: "Tensor",
    doc: [
      "exp(input, *, out=None) -&gt; Tensor  ",
      "  ",
      "Returns a new tensor with the exponential of the elements  ",
      "of the input tensor :attr:`input`.  ",
      "  ",
      "```````````python",
      "    y_{i} = e^{x_{i}}",
      "```````````",
      "",
      "## Arguments",
      "**input**: `Tensor`  ",
      "the input tensor.",
      "",
      "## Keyword Arguments",
      "**out**: `Tensor, optional`  ",
      "the output tensor.",
      "",
      "Example:    ",
      "```````````python",
      "    >>> torch.exp(torch.tensor([0, math.log(2.)]))",
      "    tensor([ 1.,  2.])",
      "```````````",
    ].join("\n"),
  },
  exp2: {
    kind: FUNCTION,
    signature: "def exp2(input: Tensor, *, out: Tensor | None = None) -> Tensor",
    parameters: ["input: Tensor", "out: Tensor | None = None"],
    result: "Tensor",
    doc: "exp2(input, *, out=None) -&gt; Tensor  \n  \nAlias for :func:`torch.special.exp2`.",
  },
  expm1: {
    kind: FUNCTION,
    signature: "def expm1(input: Tensor, *, out: Tensor | None = None) -> Tensor",
    parameters: ["input: Tensor", "out: Tensor | None = None"],
    result: "Tensor",
    doc: "Returns a new tensor with the exponential of the elements minus 1 of :attr:`input`.",
  },
  export: { kind: MODULE, signature: "<module 'torch.export'>", doc: null },
  linalg: { kind: MODULE, signature: "<module 'torch.linalg'>", doc: null },
  nn: { kind: MODULE, signature: "<module 'torch.nn'>", doc: null },
  logsumexp: {
    kind: FUNCTION,
    signature:
      "def logsumexp(input: Tensor, dim: int | tuple[int, ...], keepdim: bool = False, *, out: Tensor | None = None) -> Tensor",
    parameters: [
      "input: Tensor",
      "dim: int | tuple[int, ...]",
      "keepdim: bool = False",
      "out: Tensor | None = None",
    ],
    result: "Tensor",
    doc: [
      "Returns the log of summed exponentials of each row of the :attr:`input`  ",
      "tensor in the given dimension :attr:`dim`. The computation is numerically  ",
      "stabilized.",
      "",
      "## Arguments",
      "**input**: `Tensor`  ",
      "the input tensor.",
      "",
      "**dim**: `int or tuple of ints`  ",
      "the dimension or dimensions to reduce.",
      "",
      "**keepdim**: `bool`  ",
      "whether the output tensor has :attr:`dim` retained or not.",
    ].join("\n"),
  },
  softmax: {
    kind: FUNCTION,
    signature: "def softmax(input: Tensor, dim: int, dtype: dtype | None = None) -> Tensor",
    parameters: ["input: Tensor", "dim: int", "dtype: dtype | None = None"],
    result: "Tensor",
    doc: "softmax(input, dim, *, dtype=None) -&gt; Tensor  \n  \nAlias for :func:`torch.nn.functional.softmax`.",
  },
  Tensor: { kind: CLASS, signature: "<class 'Tensor'>", doc: null },
  tensor: {
    kind: FUNCTION,
    signature: "def tensor(data: Any, dtype: dtype | None = None) -> Tensor",
    parameters: ["data: Any", "dtype: dtype | None = None"],
    result: "Tensor",
    doc: "Constructs a tensor with no autograd history by copying :attr:`data`.",
  },
};

/** Members the server knows only by signature; their docstrings exist at import time (the docs helper). */
const FUNCTIONAL: Record<string, Member> = {
  scaled_dot_product_attention: {
    kind: FUNCTION,
    signature:
      "def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None, dropout_p: float = ..., is_causal: bool = False) -> Tensor",
    parameters: [
      "query: Tensor",
      "key: Tensor",
      "value: Tensor",
      "attn_mask: Tensor | None = None",
      "dropout_p: float = ...",
      "is_causal: bool = False",
    ],
    result: "Tensor",
    doc: null,
  },
  softmax: {
    kind: FUNCTION,
    signature: "def softmax(input: Tensor, dim: int | None = None, dtype: dtype | None = None) -> Tensor",
    parameters: ["input: Tensor", "dim: int | None = None", "dtype: dtype | None = None"],
    result: "Tensor",
    doc: null,
  },
};

const RUNTIME_DOCS: Record<string, { signature: string; doc: string }> = {
  "torch.nn.functional.scaled_dot_product_attention": {
    signature:
      "scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> Tensor",
    doc: [
      "Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,",
      "and applying dropout if a probability greater than 0.0 is specified.",
      "",
      ".. math::",
      "    \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right) V",
      "",
      "Args:",
      "    query (Tensor): Query tensor; shape `(N, ..., Hq, L, E)`.",
      "    key (Tensor): Key tensor; shape `(N, ..., H, S, E)`.",
      "    value (Tensor): Value tensor; shape `(N, ..., H, S, Ev)`.",
      "    attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights.",
      "    is_causal (bool): If set to true, the attention masking is a lower triangular matrix.",
      "",
      ".. note::",
      "    Dropout is always applied according to `dropout_p`, whatever the module's training mode.",
      "",
      "Examples::",
      "",
      "    >>> query = torch.rand(32, 8, 128, 64)",
      "    >>> F.scaled_dot_product_attention(query, key, value)",
    ].join("\n"),
  },
  "torch.nn.functional.softmax": {
    signature: "softmax(input, dim=None, _stacklevel=3, dtype=None) -> Tensor",
    doc: [
      "Apply a softmax function.",
      "",
      "Softmax is defined as:",
      "",
      "`\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}`",
      "",
      "Args:",
      "    input (Tensor): input",
      "    dim (int): A dimension along which softmax will be computed.",
    ].join("\n"),
  },
};

interface Namespace {
  /** The public dotted path. */
  path: string;
  members: Record<string, Member>;
}

const NAMESPACES: Record<string, Namespace> = {
  torch: { path: "torch", members: TORCH },
  "torch.nn.functional": { path: "torch.nn.functional", members: FUNCTIONAL },
};

/** What each name at the start of a dotted expression refers to, from the file's imports. */
function importsOf(source: string): Map<string, string> {
  const names = new Map<string, string>();
  for (const line of source.split("\n")) {
    const plain = /^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?/.exec(line);
    if (plain?.[1]) {
      if (plain[2]) names.set(plain[2], plain[1]);
      else names.set(plain[1].split(".")[0] ?? plain[1], plain[1].split(".")[0] ?? plain[1]);
    }
    const from = /^\s*from\s+([\w.]+)\s+import\s+(\w+)(?:\s+as\s+(\w+))?/.exec(line);
    if (from?.[1] && from[2]) names.set(from[3] ?? from[2], `${from[1]}.${from[2]}`);
  }
  return names;
}

/** The namespace a receiver (`torch`, `F`, `torch.nn.functional`) names in this file, if the fixture knows it. */
function namespaceOf(source: string, receiver: string): Namespace | null {
  const [head = "", ...rest] = receiver.split(".");
  const base = importsOf(source).get(head);
  if (!base) return null;
  return NAMESPACES[[base, ...rest].join(".")] ?? null;
}

interface Position {
  line: number;
  character: number;
}

function offsetAt(text: string, position: Position): number {
  const lines = text.split("\n");
  let offset = 0;
  for (let index = 0; index < position.line && index < lines.length; index += 1) {
    offset += (lines[index] ?? "").length + 1;
  }
  return offset + position.character;
}

function positionAt(text: string, offset: number): Position {
  const before = text.slice(0, offset).split("\n");
  return { line: before.length - 1, character: (before[before.length - 1] ?? "").length };
}

/** The `receiver.member` reference around an offset, if there is one. */
function referenceAt(
  text: string,
  offset: number,
): { receiver: string; member: string; from: number; to: number } | null {
  let start = offset;
  let end = offset;
  while (start > 0 && /\w/.test(text[start - 1] ?? "")) start -= 1;
  while (end < text.length && /\w/.test(text[end] ?? "")) end += 1;
  const receiver = /([A-Za-z_][\w.]*)\.$/.exec(text.slice(0, start));
  if (!receiver?.[1]) return null;
  return { receiver: receiver[1], member: text.slice(start, end), from: start, to: end };
}

/** True when `b` is `a` with one character inserted, removed, or replaced. */
function oneEditApart(a: string, b: string): boolean {
  if (Math.abs(a.length - b.length) > 1 || a === b) return false;
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA -= 1;
    endB -= 1;
  }
  return endA - start <= 1 && endB - start <= 1;
}

function signatureLabel(member: Member): string {
  return `(${(member.parameters ?? []).join(", ")}) -> ${member.result ?? "Unknown"}`;
}

type Scenario = "editor_restarting" | "editor_failed" | "editor_not_installed";
const SCENARIOS: ReadonlySet<string> = new Set<Scenario>([
  "editor_restarting",
  "editor_failed",
  "editor_not_installed",
]);

interface Session {
  id: string;
  generation: number;
  emit: (event: EditorEvent) => void;
  /** Open documents of the current server process, by URI. */
  documents: Map<string, string>;
  played: Set<Scenario>;
  running: boolean;
  ended: boolean;
}

export interface FixtureEditorOptions {
  sleep: (ms: number) => Promise<void>;
  /** Whether the language server starts out installed. */
  installed?: boolean;
}

export class FixtureEditor {
  enabled = true;
  private installed: boolean;
  private installing = false;
  /** True after an install: `editor_not_installed` plays until then, not again afterwards. */
  private reinstalled = false;
  private session: Session | null = null;
  private latest: EditorState | null = null;
  private nextSession = 1;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(options: FixtureEditorOptions) {
    this.sleep = options.sleep;
    this.installed = options.installed ?? true;
  }

  status(): EditorStatus {
    return {
      enabled: this.enabled,
      installed: this.installed,
      installing: this.installing,
      serverVersion: this.installed ? SERVER_VERSION : null,
      state: this.latest,
    };
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) this.end(this.session, { status: "stopped", reason: "disabled" });
  }

  start(onEvent: (event: EditorEvent) => void): string {
    if (!this.enabled) throw new Error("editor intelligence is turned off in Settings");
    this.end(this.session, { status: "stopped", reason: "replaced" });
    const session: Session = {
      id: `fixture-editor-${this.nextSession++}`,
      generation: 0,
      emit: onEvent,
      documents: new Map(),
      played: new Set(),
      running: false,
      ended: false,
    };
    this.session = session;
    void (async () => {
      await this.sleep(20);
      if (!this.installed) {
        this.end(session, {
          status: "notInstalled",
          message: "The language server is not installed. Install it from Settings.",
        });
        return;
      }
      this.state(session, { status: "starting" });
      await this.sleep(150);
      this.ready(session);
    })();
    return session.id;
  }

  stop(sessionId?: string): void {
    if (sessionId !== undefined && this.session?.id !== sessionId) return;
    this.end(this.session, { status: "stopped", reason: "requested" });
  }

  send(sessionId: string, message: string): void {
    const session = this.session;
    if (!session || session.id !== sessionId || session.ended || !session.running) {
      throw new Error("the editor session is not running");
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(message);
    } catch {
      parsed = null;
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("invalid argument: the message is not a JSON object");
    }
    const generation = session.generation;
    // A real server answers on another thread; never inside the call that sent the request.
    void this.sleep(15).then(() => {
      if (session.ended || session.generation !== generation) return;
      this.handle(session, parsed as { id?: number | string; method?: string; params?: unknown });
    });
  }

  async install(onEvent: (event: InstallEvent) => void): Promise<void> {
    if (this.installing) throw new Error("An install is already running.");
    this.installing = true;
    const lines = [
      "$ uv sync --project python --extra torch --extra jax --extra editor",
      "Resolved 62 packages in 3ms",
      "Downloading ty (14.1MiB)",
      "Prepared 1 package in 1.12s",
      "Installed 1 package in 9ms",
      ` + ty==${SERVER_VERSION}`,
    ];
    void (async () => {
      for (const text of lines) {
        await this.sleep(260);
        onEvent({ type: "log", text });
      }
      await this.sleep(200);
      this.installed = true;
      this.reinstalled = true;
      this.installing = false;
      onEvent({ type: "done", ok: true, message: null });
    })();
  }

  docs(source: string, expr: string, member?: string): EditorDocs {
    if (!this.enabled) throw new Error("editor intelligence is turned off in Settings");
    const missing = (reason: string): EditorDocs => ({
      found: false,
      name: null,
      kind: null,
      signature: null,
      doc: null,
      truncated: false,
      reason,
    });
    const dotted = member === undefined ? expr : expr === "" ? member : `${expr}.${member}`;
    if (!/^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$/.test(dotted)) return missing("not_a_name");
    const [head = "", ...rest] = dotted.split(".");
    const base = importsOf(source).get(head) ?? (head === "torch" ? "torch" : null);
    if (!base) return missing("unknown_name");
    const name = [base, ...rest].join(".");
    const known = RUNTIME_DOCS[name];
    if (known) {
      return { found: true, name, kind: "function", ...known, truncated: false, reason: null };
    }
    const last = rest[rest.length - 1];
    const namespace = NAMESPACES[[base, ...rest.slice(0, -1)].join(".")];
    if (last && namespace?.members[last]) {
      return {
        found: true,
        name,
        kind: "function",
        signature: null,
        doc: null,
        truncated: false,
        reason: null,
      };
    }
    return missing("no_attribute");
  }

  // The server ----------------------------------------------------------------

  private state(session: Session, state: EditorState): void {
    if (session.ended) return;
    if (this.session === session) this.latest = state;
    session.emit({ type: "state", state });
  }

  private end(session: Session | null, state: EditorState): void {
    if (!session || session.ended) return;
    this.state(session, state);
    session.ended = true;
    session.running = false;
    if (this.session === session) this.session = null;
  }

  private ready(session: Session): void {
    if (session.ended) return;
    session.generation += 1;
    session.documents.clear();
    session.running = true;
    this.state(session, {
      status: "ready",
      sessionId: session.id,
      serverName: "ty",
      serverVersion: SERVER_VERSION,
      rootUri: ROOT_URI,
      workspacePath: ROOT_URI.replace("file://", ""),
      generation: session.generation,
    });
  }

  private reply(session: Session, id: number | string, result: unknown): void {
    session.emit({ type: "message", message: JSON.stringify({ jsonrpc: "2.0", id, result }) });
  }

  private handle(session: Session, message: { id?: number | string; method?: string; params?: unknown }) {
    const params = (message.params ?? {}) as {
      textDocument?: { uri?: string; text?: string; version?: number };
      contentChanges?: { text: string }[];
      position?: Position;
    };
    const uri = params.textDocument?.uri ?? "";
    const text = session.documents.get(uri) ?? "";
    const { id } = message;
    switch (message.method) {
      case "initialize":
        if (id !== undefined) {
          this.reply(session, id, {
            capabilities: {
              // 1: the whole document with every change.
              textDocumentSync: { openClose: true, change: 1 },
              completionProvider: { triggerCharacters: ["."] },
              hoverProvider: true,
              signatureHelpProvider: { triggerCharacters: ["(", ","], retriggerCharacters: [")"] },
              positionEncoding: "utf-16",
            },
            serverInfo: { name: "ty", version: SERVER_VERSION },
          });
        }
        return;
      case "textDocument/didOpen":
        this.opened(session, uri, params.textDocument?.text ?? "", params.textDocument?.version ?? 0);
        return;
      case "textDocument/didChange": {
        const change = params.contentChanges?.[params.contentChanges.length - 1];
        if (change) this.opened(session, uri, change.text, params.textDocument?.version ?? 0);
        return;
      }
      case "textDocument/didClose":
        session.documents.delete(uri);
        return;
      case "textDocument/completion":
        if (id !== undefined) this.reply(session, id, this.completion(text, params.position));
        return;
      case "textDocument/hover":
        if (id !== undefined) this.reply(session, id, this.hover(text, params.position));
        return;
      case "textDocument/signatureHelp":
        if (id !== undefined) this.reply(session, id, this.signatureHelp(text, params.position));
        return;
      case "shutdown":
        if (id !== undefined) this.reply(session, id, null);
        return;
      default:
        if (id !== undefined && message.method !== undefined) {
          session.emit({
            type: "message",
            message: JSON.stringify({
              jsonrpc: "2.0",
              id,
              error: { code: -32601, message: `Method not found: ${message.method}` },
            }),
          });
        }
    }
  }

  private opened(session: Session, uri: string, text: string, version: number): void {
    session.documents.set(uri, text);
    for (const marker of text.matchAll(/#\s*learnai:\s*(editor_[a-z_]+)/g)) {
      const scenario = marker[1] as Scenario;
      if (!SCENARIOS.has(scenario) || session.played.has(scenario)) continue;
      // Once the server has been installed from Settings, the marker has had its say.
      if (scenario === "editor_not_installed" && this.reinstalled) continue;
      session.played.add(scenario);
      this.play(session, scenario);
      return;
    }
    session.emit({
      type: "message",
      message: JSON.stringify({
        jsonrpc: "2.0",
        method: "textDocument/publishDiagnostics",
        params: { uri, version, diagnostics: this.diagnostics(text) },
      }),
    });
  }

  /** Plays a session scenario. Each one takes the server process away, for a while or for good. */
  private play(session: Session, scenario: Scenario): void {
    if (scenario === "editor_not_installed") {
      this.installed = false;
      this.end(session, {
        status: "notInstalled",
        message: "The language server is not installed. Install it from Settings.",
      });
      return;
    }
    session.running = false;
    this.state(session, {
      status: "restarting",
      attempt: 1,
      maxAttempts: 4,
      delayMs: 500,
      message: "The language server stopped unexpectedly. Starting it again.",
    });
    void (async () => {
      await this.sleep(1500);
      if (scenario === "editor_failed") {
        this.end(session, {
          status: "failed",
          message:
            "The language server keeps stopping. Suggestions are off until you open another problem or restart Learn AI.",
        });
        return;
      }
      this.state(session, { status: "starting" });
      await this.sleep(300);
      this.ready(session);
    })();
  }

  private diagnostics(text: string): unknown[] {
    const found: unknown[] = [];
    for (const match of text.matchAll(/\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)/g)) {
      const namespace = namespaceOf(text, match[1] ?? "");
      const member = match[2] ?? "";
      if (!namespace || member in namespace.members || namespace.path !== "torch") continue;
      // The fixture knows a handful of members, so only a near miss of one counts as a mistake.
      if (!Object.keys(namespace.members).some((known) => oneEditApart(known, member))) continue;
      // `import torch.nn.functional as F` names submodules, not members.
      const lineStart = text.lastIndexOf("\n", match.index) + 1;
      if (/^\s*(import|from)\s/.test(text.slice(lineStart, match.index))) continue;
      found.push({
        range: {
          start: positionAt(text, match.index),
          end: positionAt(text, match.index + match[0].length),
        },
        severity: 1,
        code: "missing-attribute",
        source: "ty",
        message: `Module \`${namespace.path}\` has no member \`${member}\``,
      });
    }
    // A hint, the way the server marks an unused name: the client is expected to drop it.
    const unused = /^\s*(unused\w*)\s*=/m.exec(text);
    if (unused?.[1]) {
      const start = unused.index + unused[0].indexOf(unused[1]);
      found.push({
        range: { start: positionAt(text, start), end: positionAt(text, start + unused[1].length) },
        severity: 4,
        source: "ty",
        tags: [1],
        message: `\`${unused[1]}\` is unused`,
      });
    }
    return found;
  }

  private completion(text: string, position?: Position): unknown {
    if (!position) return null;
    const reference = referenceAt(text, offsetAt(text, position));
    if (!reference) {
      const names = [...importsOf(text).keys()];
      return {
        isIncomplete: false,
        items: names.map((label) => ({ label, kind: MODULE, detail: `<module '${label}'>` })),
      };
    }
    const namespace = namespaceOf(text, reference.receiver);
    if (!namespace) return { isIncomplete: false, items: [] };
    // The real server ranks modules first; the client is expected to put the best match on top.
    const entries = Object.entries(namespace.members).sort(
      ([, a], [, b]) => Number(b.kind === MODULE) - Number(a.kind === MODULE),
    );
    return {
      isIncomplete: false,
      items: entries.map(([label, member], index) => ({
        label,
        kind: member.kind,
        detail: member.signature,
        sortText: String(index).padStart(2, " "),
        ...(member.doc ? { documentation: { kind: "markdown", value: member.doc } } : {}),
      })),
    };
  }

  private hover(text: string, position?: Position): unknown {
    if (!position) return null;
    const reference = referenceAt(text, offsetAt(text, position));
    const member = reference && namespaceOf(text, reference.receiver)?.members[reference.member];
    if (!reference || !member) return null;
    const language = member.kind === MODULE ? "xml" : "python";
    const code = member.signature.replace(/\((.+)\)/, (_all, inner: string) =>
      inner.length > 40 ? `(\n    ${inner.split(", ").join(",\n    ")}\n)` : `(${inner})`,
    );
    return {
      contents: {
        kind: "markdown",
        value: `\`\`\`${language}\n${code}\n\`\`\`${member.doc ? `\n---\n${member.doc}` : ""}`,
      },
      range: { start: positionAt(text, reference.from), end: positionAt(text, reference.to) },
    };
  }

  private signatureHelp(text: string, position?: Position): unknown {
    if (!position) return null;
    const offset = offsetAt(text, position);
    // Walk back to the parenthesis this position is inside, counting the commas on the way.
    let depth = 0;
    let commas = 0;
    let open = -1;
    for (let index = offset - 1; index >= 0; index -= 1) {
      const char = text[index];
      if (char === "\n") break;
      if (char === ")") depth += 1;
      else if (char === "(") {
        if (depth === 0) {
          open = index;
          break;
        }
        depth -= 1;
      } else if (char === "," && depth === 0) commas += 1;
    }
    if (open < 0) return null;
    const reference = referenceAt(text, open);
    const member = reference && namespaceOf(text, reference.receiver)?.members[reference.member];
    if (!member?.parameters) return null;
    const label = signatureLabel(member);
    let cursor = 1;
    const parameters = member.parameters.map((parameter) => {
      const start = label.indexOf(parameter, cursor);
      cursor = start + parameter.length;
      return {
        label: [start, cursor],
        ...(parameter.startsWith("input") ? { documentation: "the input tensor." } : {}),
      };
    });
    const activeParameter = Math.min(commas, parameters.length - 1);
    return {
      activeSignature: 0,
      activeParameter,
      signatures: [{ label, parameters, activeParameter }],
    };
  }
}
