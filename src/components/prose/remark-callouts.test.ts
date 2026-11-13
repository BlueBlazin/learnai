import { describe, expect, it } from "vitest";
import { remarkCallouts } from "./remark-callouts";

const text = (value: string) => ({ type: "text", value });
const paragraph = (...children: unknown[]) => ({ type: "paragraph", children });

describe("remarkCallouts", () => {
  it("turns a marked blockquote into a callout and strips the marker", () => {
    const tree = {
      type: "root",
      children: [{ type: "blockquote", children: [paragraph(text("[!WARNING]\nMind the scale."))] }],
    };
    remarkCallouts()(tree);
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
    remarkCallouts()(tree);
    expect((tree.children[0] as { data?: unknown }).data).toBeUndefined();
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
    remarkCallouts()(tree);
    remarkCallouts()(tree);
    const children = (
      tree.children[0] as { children: { type: string; value?: string; children?: { value?: string }[] }[] }
    ).children;
    expect(children.map((c) => c.type)).toEqual(["mathWithPunctuation", "text", "inlineMath", "text"]);
    expect(children[0]?.children?.[1]?.value).toBe(",");
    expect(children[1]?.value).toBe(" it computes");
  });
});
