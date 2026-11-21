import { describe, expect, it } from "vitest";
import { INTERACTIVE_FIGURES, interactiveFigure, interactiveNames } from "./registry";

/** Every Markdown file of the shipped problems (statements, hints, editorials), by path. */
const PROBLEM_TEXTS = import.meta.glob<string>("../../../problems/**/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
});

describe("interactive figures", () => {
  it("names every figure in lowercase words joined by hyphens", () => {
    expect(Object.keys(INTERACTIVE_FIGURES).sort()).toEqual([
      "quant-grid",
      "quant-grid-affine",
      "rope-shift",
      "sampler-order",
    ]);
    for (const name of Object.keys(INTERACTIVE_FIGURES)) expect(name).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
  });

  it("looks a figure up by its own name only", () => {
    expect(interactiveFigure("sampler-order")).not.toBeNull();
    expect(interactiveFigure("no-such-figure")).toBeNull();
    expect(interactiveFigure("toString")).toBeNull();
    expect(interactiveFigure("")).toBeNull();
  });

  it("finds the names a Markdown text uses", () => {
    const text = [
      "![Ten tokens, $p$ and all.](interactive:sampler-order)",
      "![A drawing](assets/mask.svg) and ![Other](interactive:kv-growth)",
      "The scheme interactive:prose is not an image.",
    ].join("\n\n");
    expect(interactiveNames(text)).toEqual(["sampler-order", "kv-growth"]);
  });

  it("has every interactive figure that a problem names", () => {
    const files = Object.entries(PROBLEM_TEXTS);
    expect(files.length).toBeGreaterThan(100);
    const unknown = files.flatMap(([file, text]) =>
      interactiveNames(text)
        .filter((name) => interactiveFigure(name) === null)
        .map((name) => `${file}: interactive:${name}`),
    );
    expect(unknown).toEqual([]);
  });
});
