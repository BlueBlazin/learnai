import { describe, expect, it } from "vitest";
import { interactiveName, remarkProse } from "./remark-prose";

const text = (value: string) => ({ type: "text", value });
const paragraph = (...children: unknown[]) => ({ type: "paragraph", children });

describe("remarkProse", () => {
  it("turns a marked blockquote into a callout and strips the marker", () => {
    const tree = {
      type: "root",
      children: [{ type: "blockquote", children: [paragraph(text("[!WARNING]\nMind the scale."))] }],
    };
    remarkProse()(tree);
    const quote = tree.children[0] as {
      data?: { hName?: string; hProperties?: Record<string, unknown> };
      children: { children: { value: string }[] }[];
    };
    expect(quote.data?.hName).toBe("aside");
    expect(quote.data?.hProperties?.["data-kind"]).toBe("warning");
    expect(quote.children[0]?.children[0]?.value).toBe("Mind the scale.");
  });

  it("leaves ordinary quotations alone", () => {
    const tree = {
      type: "root",
      children: [{ type: "blockquote", children: [paragraph(text("Just a quote."))] }],
    };
    remarkProse()(tree);
    expect((tree.children[0] as { data?: unknown }).data).toBeUndefined();
  });

  it("turns a lone image into a numbered figure, and marks an interactive one by name", () => {
    const image = (url: string, alt: string) => ({ type: "image", url, alt });
    const tree = {
      type: "root",
      children: [
        paragraph(image("assets/mask.svg", "The mask.")),
        paragraph(text("See "), image("interactive:sampler-order", "inline"), text(".")),
        paragraph(image("interactive:sampler-order", "Ten tokens.")),
        paragraph(image("interactive:", "No name.")),
      ],
    };
    remarkProse()(tree);
    const data = tree.children.map(
      (node) => (node as { data?: { hName?: string; hProperties?: Record<string, unknown> } }).data,
    );
    expect(data[0]?.hName).toBe("figure");
    expect(data[0]?.hProperties).toEqual({
      "data-src": "assets/mask.svg",
      "data-number": 1,
      "data-alt": "The mask.",
      "data-caption": "The mask.",
    });
    // An image inside a sentence is not a figure, whatever its URL.
    expect(data[1]).toBeUndefined();
    expect(data[2]?.hProperties).toEqual({
      "data-src": "interactive:sampler-order",
      "data-number": 2,
      "data-alt": "Ten tokens.",
      "data-caption": "Ten tokens.",
      "data-interactive": "sampler-order",
    });
    expect(data[3]?.hProperties?.["data-interactive"]).toBe("");
    expect(data[3]?.hProperties?.["data-number"]).toBe(3);
  });

  it("reads the name of an interactive figure from its URL", () => {
    expect(interactiveName("interactive:sampler-order")).toBe("sampler-order");
    expect(interactiveName("assets/interactive:x.svg")).toBeNull();
    expect(interactiveName("https://example.com/interactive:x")).toBeNull();
  });

  it("binds punctuation to preceding inline math exactly once", () => {
    const tree = {
      type: "root",
      children: [
        paragraph(
          { type: "inlineMath", value: "d_v" },
          text(", it computes"),
          { type: "inlineMath", value: "x" },
          text(" and more"),
        ),
      ],
    };
    remarkProse()(tree);
    remarkProse()(tree);
    const children = (
      tree.children[0] as { children: { type: string; value?: string; children?: { value?: string }[] }[] }
    ).children;
    expect(children.map((c) => c.type)).toEqual(["mathWithPunctuation", "text", "inlineMath", "text"]);
    expect(children[0]?.children?.[1]?.value).toBe(",");
    expect(children[1]?.value).toBe(" it computes");
  });
});
