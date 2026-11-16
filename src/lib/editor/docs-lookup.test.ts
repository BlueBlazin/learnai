import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EditorDocs } from "@/lib/api/types";
import { DocsLookup, importSignature, mayBeLibraryName, needsDocsFallback } from "./docs-lookup";

const found = (name: string, doc: string | null = "Docs."): EditorDocs => ({
  found: true,
  name,
  kind: "function",
  signature: `${name}()`,
  doc,
  truncated: false,
  reason: null,
});
const missing = (reason: string): EditorDocs => ({
  found: false,
  name: null,
  kind: null,
  signature: null,
  doc: null,
  truncated: false,
  reason,
});

const SOURCE = "import torch\nimport torch.nn.functional as F\n\nx = 1\n";

describe("needsDocsFallback", () => {
  it("asks the docs helper exactly when the server gave no documentation", () => {
    expect(needsDocsFallback(null)).toBe(true);
    expect(needsDocsFallback(undefined)).toBe(true);
    expect(needsDocsFallback("")).toBe(true);
    expect(needsDocsFallback("  \n ")).toBe(true);
    expect(needsDocsFallback("Returns a new tensor.")).toBe(false);
  });
});

describe("mayBeLibraryName", () => {
  it("accepts names that start at an import of the file or at a known package", () => {
    expect(mayBeLibraryName(SOURCE, "F.scaled_dot_product_attention")).toBe(true);
    expect(mayBeLibraryName(SOURCE, "torch.linalg.svd")).toBe(true);
    expect(mayBeLibraryName("from torch import nn, Tensor as T", "nn.Linear")).toBe(true);
    expect(mayBeLibraryName("from torch import nn, Tensor as T", "T")).toBe(true);
    expect(mayBeLibraryName("import numpy as np, math", "math.pi")).toBe(true);
    expect(mayBeLibraryName("", "jax.numpy.exp")).toBe(true);
  });

  it("rejects local names, and gives imports it cannot read the benefit of the doubt", () => {
    expect(mayBeLibraryName(SOURCE, "x")).toBe(false);
    expect(mayBeLibraryName(SOURCE, "logits.max")).toBe(false);
    expect(mayBeLibraryName("from torch import nn, Tensor as T", "Tensor")).toBe(false);
    expect(mayBeLibraryName("from torch import (\n    nn,\n)", "nn.Linear")).toBe(true);
    expect(mayBeLibraryName("from math import *", "pi")).toBe(true);
  });
});

describe("DocsLookup", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("debounces: moving on before the pause passes over the earlier name", async () => {
    const fetchDocs = vi.fn(async (_source: string, expr: string, member?: string) =>
      found(member ? `${expr}.${member}` : expr),
    );
    const lookup = new DocsLookup(fetchDocs, { debounceMs: 100 });
    const first = lookup.lookup(SOURCE, "torch", "exp");
    await vi.advanceTimersByTimeAsync(60);
    const second = lookup.lookup(SOURCE, "torch", "expm1");
    await vi.advanceTimersByTimeAsync(100);
    await expect(first).resolves.toBeNull();
    await expect(second).resolves.toMatchObject({ name: "torch.expm1" });
    expect(fetchDocs).toHaveBeenCalledTimes(1);
    expect(fetchDocs).toHaveBeenCalledWith(SOURCE, "torch", "expm1");
    // The name that was passed over was never asked, so it is not remembered as "nothing".
    expect(lookup.peek(SOURCE, "torch", "exp")).toBeUndefined();
  });

  it("caches answers, including 'nothing', and shares a request in flight", async () => {
    const fetchDocs = vi.fn(async (_source: string, expr: string) =>
      expr === "x" ? missing("unknown_name") : found(expr),
    );
    const lookup = new DocsLookup(fetchDocs, { debounceMs: 10 });
    const a = lookup.lookup(SOURCE, "F.softmax");
    const b = lookup.lookup(SOURCE, "F.softmax");
    await vi.advanceTimersByTimeAsync(10);
    expect(await a).toMatchObject({ name: "F.softmax" });
    expect(await b).toBe(await a);
    // Cached answers come back at once, with no debounce and no call.
    await expect(lookup.lookup(SOURCE, "F.softmax")).resolves.toMatchObject({ name: "F.softmax" });

    const none = lookup.lookup(SOURCE, "x");
    await vi.advanceTimersByTimeAsync(10);
    await expect(none).resolves.toBeNull();
    await expect(lookup.lookup(SOURCE, "x")).resolves.toBeNull();
    expect(lookup.peek(SOURCE, "x")).toBeNull();
    expect(fetchDocs).toHaveBeenCalledTimes(2);
  });

  it("keys the cache on the imports, not on the rest of the file", async () => {
    const fetchDocs = vi.fn(async (_source: string, expr: string) => found(expr));
    const lookup = new DocsLookup(fetchDocs, { debounceMs: 0 });
    const first = lookup.lookup(SOURCE, "F.softmax");
    await vi.advanceTimersByTimeAsync(0);
    await first;
    await lookup.lookup(`${SOURCE}\ny = F.softmax(x)\n`, "F.softmax");
    expect(fetchDocs).toHaveBeenCalledTimes(1);

    const other = lookup.lookup("import jax.nn as F\n", "F.softmax");
    await vi.advanceTimersByTimeAsync(0);
    await other;
    expect(fetchDocs).toHaveBeenCalledTimes(2);
    expect(importSignature(`${SOURCE}\n    import math\n`)).toBe(
      "import torch\nimport torch.nn.functional as F\nimport math",
    );
  });

  it("never rejects and does not remember a failure or a busy helper", async () => {
    const fetchDocs = vi
      .fn<(source: string, expr: string, member?: string) => Promise<EditorDocs>>()
      .mockRejectedValueOnce(new Error("editor intelligence is turned off in Settings"))
      .mockResolvedValueOnce(missing("unavailable"))
      .mockResolvedValueOnce(found("torch.exp"));
    const lookup = new DocsLookup(fetchDocs, { debounceMs: 0 });
    for (const expected of [null, null, "torch.exp"]) {
      const answer = lookup.lookup(SOURCE, "torch.exp");
      await vi.advanceTimersByTimeAsync(0);
      expect((await answer)?.name ?? null).toBe(expected);
    }
    expect(fetchDocs).toHaveBeenCalledTimes(3);
  });

  it("treats a name with neither a docstring nor a signature as nothing to show", async () => {
    const bare: EditorDocs = { ...found("torch.pi", null), signature: null, kind: "value" };
    const lookup = new DocsLookup(async () => bare, { debounceMs: 0 });
    const answer = lookup.lookup(SOURCE, "torch.pi");
    await vi.advanceTimersByTimeAsync(0);
    await expect(answer).resolves.toBeNull();
  });

  it("forgets the oldest answers past its limit, and everything on clear", async () => {
    const fetchDocs = vi.fn(async (_source: string, expr: string) => found(expr));
    const lookup = new DocsLookup(fetchDocs, { debounceMs: 0, maxEntries: 2 });
    for (const name of ["torch.a", "torch.b", "torch.c"]) {
      const answer = lookup.lookup(SOURCE, name);
      await vi.advanceTimersByTimeAsync(0);
      await answer;
    }
    expect(lookup.peek(SOURCE, "torch.a")).toBeUndefined();
    expect(lookup.peek(SOURCE, "torch.c")).toMatchObject({ name: "torch.c" });
    lookup.clear();
    expect(lookup.peek(SOURCE, "torch.c")).toBeUndefined();
  });
});
