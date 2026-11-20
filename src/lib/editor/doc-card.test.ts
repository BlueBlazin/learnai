import { describe, expect, it } from "vitest";
import type { EditorDocs } from "@/lib/api/types";
import { completionContent, hasDocContent, hoverContent, hoverNeedsDocs, serverItemDocs } from "./doc-card";
import { inlineText, parseHover } from "./doc-format";

const HELPER: EditorDocs = {
  found: true,
  name: "torch.nn.functional.scaled_dot_product_attention",
  kind: "function",
  signature: "scaled_dot_product_attention(query, key, value) -> Tensor",
  doc: "Computes scaled dot product attention.\n\nArgs:\n    query (Tensor): Query tensor.",
  truncated: true,
  reason: null,
};

const text = (content: ReturnType<typeof hoverContent>) =>
  (content.blocks ?? []).map((block) => ("content" in block ? inlineText(block.content) : block.kind));

/** What the server sends as the documentation of a name whose type it could not infer. */
const UNKNOWN_DOC =
  "`Unknown` is a dynamic type inferred due to missing type information or an inference error.";

describe("hoverContent", () => {
  it("shows what the server gave when it has a docstring, with the helper's name at most", () => {
    const hover = parseHover(
      "```python\ndef exp(\n    input: Tensor\n) -> Tensor\n```\n---\nReturns the exponential.",
    );
    const content = hoverContent(hover, null);
    expect(content.signature).toBe("def exp(input: Tensor) -> Tensor");
    expect(text(content)).toEqual(["Returns the exponential."]);
    expect(content.caption).toBeNull();
    expect(content.truncated).toBe(false);
    // Even with a helper answer at hand, the server's documentation is the one shown.
    expect(text(hoverContent(hover, HELPER))).toEqual(["Returns the exponential."]);
  });

  it("fills in the helper's docstring under the server's signature when the server has none", () => {
    const hover = parseHover(
      "```python\ndef scaled_dot_product_attention(\n    query: Tensor\n) -> Tensor\n```",
    );
    const before = hoverContent(hover, null);
    expect(before.signature).toBe("def scaled_dot_product_attention(query: Tensor) -> Tensor");
    expect(before.blocks).toEqual([]);
    expect(hasDocContent(before)).toBe(true);

    const after = hoverContent(hover, HELPER);
    expect(after.signature).toBe(before.signature);
    expect(after.caption).toBe("torch.nn.functional.scaled_dot_product_attention");
    expect(text(after)).toEqual(["Computes scaled dot product attention.", "heading", "fields"]);
    expect(after.truncated).toBe(true);
  });

  it("uses the helper's signature when the server only says Unknown (torch.linalg)", () => {
    const hover = parseHover("```python\nUnknown\n```");
    expect(hasDocContent(hoverContent(hover, null))).toBe(false);
    const content = hoverContent(hover, HELPER);
    expect(content.signature).toBe(HELPER.signature);
    expect(hasDocContent(content)).toBe(true);
    // No hover at all, but the helper knows the name.
    expect(hoverContent(null, HELPER).signature).toBe(HELPER.signature);
    expect(hasDocContent(hoverContent(null, null))).toBe(false);
  });

  it("drops the placeholder type's own docstring from an Unknown hover", () => {
    const hover = parseHover(`\`\`\`python\nUnknown\n\`\`\`\n---\n${UNKNOWN_DOC}`);
    expect(hoverNeedsDocs(hover)).toBe(true);
    expect(hasDocContent(hoverContent(hover, null))).toBe(false);
    const content = hoverContent(hover, HELPER);
    expect(content.signature).toBe(HELPER.signature);
    expect(text(content)[0]).toBe("Computes scaled dot product attention.");
    // A real signature keeps the server's documentation.
    expect(hoverNeedsDocs(parseHover("```python\ndef exp(x)\n```\n---\nReturns the exponential."))).toBe(
      false,
    );
    expect(hoverNeedsDocs(parseHover("```python\ndef exp(x)\n```"))).toBe(true);
  });
});

describe("completionContent", () => {
  it("prefers the item's own documentation", () => {
    const content = completionContent(
      "def exp(input: Tensor) -> Tensor",
      "exp(input) -&gt; Tensor  \n  \nReturns the exponential.",
      HELPER,
    );
    expect(content.signature).toBe("def exp(input: Tensor) -> Tensor");
    expect(text(content)).toEqual(["Returns the exponential."]);
    expect(content.caption).toBeUndefined();
  });

  it("falls back to the helper for an item without documentation", () => {
    const content = completionContent(
      "def scaled_dot_product_attention(query: Tensor) -> Tensor",
      null,
      HELPER,
    );
    expect(content.signature).toBe("def scaled_dot_product_attention(query: Tensor) -> Tensor");
    expect(content.caption).toBe(HELPER.name);
    expect(text(content)[0]).toBe("Computes scaled dot product attention.");
    expect(completionContent(null, null, HELPER).signature).toBe(HELPER.signature);
  });

  it("asks the helper when the server only describes the Unknown type (torch.special.erf)", () => {
    expect(serverItemDocs("Unknown", UNKNOWN_DOC)).toEqual({ detail: null, documentation: null });
    expect(serverItemDocs("Tensor", "A tensor.")).toEqual({ detail: "Tensor", documentation: "A tensor." });
    expect(serverItemDocs(null, "A module.")).toEqual({ detail: null, documentation: "A module." });

    const content = completionContent("Unknown", UNKNOWN_DOC, HELPER);
    expect(content.signature).toBe(HELPER.signature);
    expect(content.caption).toBe(HELPER.name);
    expect(text(content)[0]).toBe("Computes scaled dot product attention.");
    // With no answer from the helper there is nothing true to say about the name.
    expect(hasDocContent(completionContent("Unknown", UNKNOWN_DOC, null))).toBe(false);
  });

  it("shows the signature alone, or nothing, when the helper has no answer", () => {
    expect(completionContent("<module 'torch.export'>", null, null)).toMatchObject({
      signature: "module torch.export",
      blocks: [],
    });
    expect(hasDocContent(completionContent(null, null, null))).toBe(false);
  });
});
