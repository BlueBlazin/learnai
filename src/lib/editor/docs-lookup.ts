import type { EditorDocs } from "@/lib/api/types";

export type FetchDocs = (source: string, expr: string, member?: string) => Promise<EditorDocs>;

export interface DocsLookupOptions {
  /** How long a lookup waits for a newer one to replace it before it is sent. */
  debounceMs?: number;
  /** How many answers are remembered. */
  maxEntries?: number;
}

/** True when the language server gave no documentation and the runtime helper should be asked. */
export function needsDocsFallback(doc: string | null | undefined): boolean {
  return doc == null || doc.trim() === "";
}

/**
 * The part of a document that decides what a dotted name refers to: its
 * `import` and `from` lines. Two documents with the same imports get the same
 * answer for the same name, so the cache survives ordinary edits.
 */
export function importSignature(source: string): string {
  return source
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^(import|from)\s/.test(line))
    .join("\n");
}

/** Packages the docs helper resolves even when the file does not import them by that name. */
const KNOWN_PACKAGES = new Set(["torch", "jax", "mlx", "numpy", "learnai"]);

/**
 * Whether a dotted name could be a library name: it starts at something the
 * file imports, or at a known package. Local variables and builtins are not,
 * and asking the helper about them is a wasted round trip. Imports this
 * cannot read on one line (`from x import (`, `import *`) count as "could be".
 */
export function mayBeLibraryName(source: string, name: string): boolean {
  const head = name.split(".")[0] ?? "";
  if (head === "") return false;
  if (KNOWN_PACKAGES.has(head)) return true;
  for (const raw of source.split("\n")) {
    const line = raw.trim();
    const plain = /^import\s+(.+)$/.exec(line);
    const from = plain ? null : /^from\s+[\w.]+\s+import\s+(.+)$/.exec(line);
    const clause = (plain?.[1] ?? from?.[1])?.replace(/#.*$/, "");
    if (clause === undefined) continue;
    if (from && /[(*\\]/.test(clause)) return true;
    for (const part of clause.split(",")) {
      const [target = "", alias] = part.trim().split(/\s+as\s+/);
      const bound = alias ?? (plain ? target.split(".")[0] : target);
      if (bound?.trim() === head) return true;
    }
  }
  return false;
}

interface Waiting {
  key: string;
  timer: ReturnType<typeof setTimeout>;
  settle: (docs: EditorDocs | null) => void;
  promise: Promise<EditorDocs | null>;
}

/**
 * Documentation from the runtime helper (`editor_docs`), for names the
 * language server has no docstring for. Lookups are debounced (moving across
 * a list of completions sends only the one that is rested on), cached, and
 * never reject: no answer, a failed call, and `found: false` are all null.
 */
export class DocsLookup {
  private readonly cache = new Map<string, EditorDocs | null>();
  private readonly inFlight = new Map<string, Promise<EditorDocs | null>>();
  private waiting: Waiting | null = null;
  private readonly debounceMs: number;
  private readonly maxEntries: number;

  constructor(
    private readonly fetchDocs: FetchDocs,
    options: DocsLookupOptions = {},
  ) {
    this.debounceMs = options.debounceMs ?? 150;
    this.maxEntries = options.maxEntries ?? 400;
  }

  private key(source: string, expr: string, member?: string): string {
    return `${expr}\u0000${member ?? ""}\u0000${importSignature(source)}`;
  }

  /** The remembered answer: the docs, null for a name that has none, undefined when not asked yet. */
  peek(source: string, expr: string, member?: string): EditorDocs | null | undefined {
    return this.cache.get(this.key(source, expr, member));
  }

  lookup(source: string, expr: string, member?: string): Promise<EditorDocs | null> {
    const key = this.key(source, expr, member);
    const cached = this.cache.get(key);
    if (cached !== undefined) return Promise.resolve(cached);
    const running = this.inFlight.get(key);
    if (running) return running;
    if (this.waiting?.key === key) return this.waiting.promise;

    // A newer lookup replaces the one still waiting: that one was passed over.
    if (this.waiting) {
      clearTimeout(this.waiting.timer);
      this.waiting.settle(null);
      this.waiting = null;
    }
    let settle: (docs: EditorDocs | null) => void = () => {};
    const promise = new Promise<EditorDocs | null>((resolve) => {
      settle = resolve;
    });
    const timer = setTimeout(() => {
      this.waiting = null;
      const request = this.request(key, source, expr, member);
      this.inFlight.set(key, request);
      void request.then((docs) => {
        this.inFlight.delete(key);
        settle(docs);
      });
    }, this.debounceMs);
    this.waiting = { key, timer, settle, promise };
    return promise;
  }

  private async request(
    key: string,
    source: string,
    expr: string,
    member?: string,
  ): Promise<EditorDocs | null> {
    let docs: EditorDocs;
    try {
      docs = await this.fetchDocs(source, expr, member);
    } catch {
      return null;
    }
    // The helper was busy or down: ask again next time instead of remembering "nothing".
    if (!docs.found && docs.reason === "unavailable") return null;
    const answer = docs.found && (docs.doc !== null || docs.signature !== null) ? docs : null;
    if (this.cache.size >= this.maxEntries) {
      const oldest = this.cache.keys().next();
      if (!oldest.done) this.cache.delete(oldest.value);
    }
    this.cache.set(key, answer);
    return answer;
  }

  /** Forgets everything: after an install, names that were missing may exist now. */
  clear(): void {
    this.cache.clear();
  }
}
