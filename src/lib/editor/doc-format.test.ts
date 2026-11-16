import { describe, expect, it } from "vitest";
import {
  type DocBlock,
  formatDoc,
  inlineText,
  isUnknownSignature,
  looksLikeTex,
  parseHover,
  parseInline,
  plainSignature,
  withoutLeadingSignature,
} from "./doc-format";

/** The hover the language server sends for `torch.exp`, verbatim. */
const SERVER_HOVER = [
  "```python",
  "def exp(",
  "    input: Tensor,",
  "    *,",
  "    out: Tensor | None = None",
  ") -> Tensor",
  "```",
  "---",
  "exp(input, *, out=None) -&gt; Tensor  ",
  "  ",
  "Returns a new tensor with the exponential of the elements  ",
  "of the input tensor :attr:`input`.  ",
  "  ",
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
].join("\n");

/** The same docstring as the runtime docs helper returns it: plain text, reStructuredText left in. */
const HELPER_DOC = [
  "Returns a new tensor with the exponential of the elements",
  "of the input tensor `input`.",
  "",
  ".. math::",
  "    y_{i} = e^{x_{i}}",
  "",
  "Args:",
  "    input (Tensor): the input tensor.",
  "",
  "Keyword args:",
  "    out (Tensor, optional): the output tensor.",
  "",
  "Example::",
  "",
  "    >>> torch.exp(torch.tensor([0, math.log(2.)]))",
  "    tensor([ 1.,  2.])",
].join("\n");

const kinds = (blocks: DocBlock[]) => blocks.map((block) => block.kind);

describe("parseHover", () => {
  it("splits the signature from the documentation", () => {
    const hover = parseHover(SERVER_HOVER);
    expect(hover.signature).toBe(
      "def exp(\n    input: Tensor,\n    *,\n    out: Tensor | None = None\n) -> Tensor",
    );
    expect(hover.doc.startsWith("exp(input, *, out=None) -> Tensor")).toBe(true);
  });

  it("reports an empty documentation when the server has no docstring", () => {
    const hover = parseHover(
      "```python\ndef scaled_dot_product_attention(\n    query: Tensor\n) -> Tensor\n```",
    );
    expect(hover.signature).toContain("scaled_dot_product_attention");
    expect(hover.doc).toBe("");
    expect(parseHover("Just prose.")).toEqual({ signature: null, doc: "Just prose." });
  });

  it("knows a hover that says nothing", () => {
    expect(isUnknownSignature(parseHover("```python\nUnknown\n```").signature)).toBe(true);
    expect(isUnknownSignature(null)).toBe(true);
    expect(isUnknownSignature("Tensor")).toBe(false);
  });
});

describe("plainSignature", () => {
  it("writes modules and classes in words", () => {
    expect(plainSignature("<module 'torch.nn'>")).toBe("module torch.nn");
    expect(plainSignature("<class 'Tensor'>")).toBe("class Tensor");
  });

  it("sets a short signature on one line and leaves a long one as the server wrote it", () => {
    expect(
      plainSignature("def exp(\n    input: Tensor,\n    *,\n    out: Tensor | None = None\n) -> Tensor"),
    ).toBe("def exp(input: Tensor, *, out: Tensor | None = None) -> Tensor");
    const long =
      "def scaled_dot_product_attention(\n    query: Tensor,\n    key: Tensor,\n    value: Tensor,\n    attn_mask: Tensor | None = None\n) -> Tensor";
    expect(plainSignature(long)).toBe(long);
    expect(plainSignature("int")).toBe("int");
  });
});

describe("withoutLeadingSignature", () => {
  it("drops a first paragraph that repeats the call signature", () => {
    expect(
      withoutLeadingSignature("exp(input, *, out=None) -&gt; Tensor  \n  \nReturns a tensor.").trim(),
    ).toBe("Returns a tensor.");
    // The server's Markdown escapes underscores.
    expect(withoutLeadingSignature("exp\\_(input) -&gt; Tensor\n\nIn-place version.").trim()).toBe(
      "In-place version.",
    );
  });

  it("keeps prose that merely starts with a call", () => {
    const text = "softmax(x) is applied to every slice\nalong dim.";
    expect(withoutLeadingSignature(text)).toBe(text);
    expect(withoutLeadingSignature("Returns a tensor.")).toBe("Returns a tensor.");
  });
});

describe("formatDoc", () => {
  it("tidies the server's Markdown: entities, roles, math, arguments, examples", () => {
    const blocks = formatDoc(withoutLeadingSignature(parseHover(SERVER_HOVER).doc));
    expect(kinds(blocks)).toEqual([
      "paragraph",
      "math",
      "heading",
      "fields",
      "heading",
      "fields",
      "heading",
      "code",
    ]);
    const [summary, math, , args, keywordHeading, , , example] = blocks;
    expect(summary?.kind === "paragraph" && summary.content).toEqual([
      {
        kind: "text",
        text: "Returns a new tensor with the exponential of the elements of the input tensor ",
      },
      { kind: "code", text: "input" },
      { kind: "text", text: "." },
    ]);
    expect(math).toEqual({ kind: "math", text: "y_{i} = e^{x_{i}}" });
    expect(args).toEqual({
      kind: "fields",
      items: [{ name: "input", type: "Tensor", content: [{ kind: "text", text: "the input tensor." }] }],
    });
    expect(keywordHeading).toEqual({ kind: "heading", text: "Keyword arguments" });
    expect(example).toEqual({
      kind: "code",
      text: ">>> torch.exp(torch.tensor([0, math.log(2.)]))\ntensor([ 1.,  2.])",
    });
  });

  it("gives the same shape for the docs helper's plain text", () => {
    const blocks = formatDoc(HELPER_DOC);
    expect(kinds(blocks)).toEqual([
      "paragraph",
      "math",
      "heading",
      "fields",
      "heading",
      "fields",
      "heading",
      "code",
    ]);
    expect(blocks[1]).toEqual({ kind: "math", text: "y_{i} = e^{x_{i}}" });
    expect(blocks[2]).toEqual({ kind: "heading", text: "Arguments" });
    expect(blocks[5]).toEqual({
      kind: "fields",
      items: [
        { name: "out", type: "Tensor, optional", content: [{ kind: "text", text: "the output tensor." }] },
      ],
    });
    expect(blocks[6]).toEqual({ kind: "heading", text: "Example" });
  });

  it("reads directives: code blocks, notes, and ones it does not know", () => {
    const blocks = formatDoc(
      [
        "Computes attention.",
        "",
        ".. code-block:: python",
        "",
        "    # Efficient implementation",
        "    def attention(q, k, v):",
        "        return q",
        "",
        ".. warning:: This function is beta.",
        "",
        ".. note::",
        "    Dropout is applied according to `dropout_p`.",
        "",
        ".. versionadded:: 2.0",
        "",
        ".. function:: max(input) -> Tensor",
        "   :noindex:",
        "",
        ".. _some-label:",
        "",
        "Done.",
      ].join("\n"),
    );
    expect(kinds(blocks)).toEqual(["paragraph", "code", "note", "note", "note", "paragraph"]);
    expect(blocks[1]).toEqual({
      kind: "code",
      text: "# Efficient implementation\ndef attention(q, k, v):\n    return q",
    });
    expect(blocks[2]).toMatchObject({ kind: "note", label: "Warning" });
    const note = blocks[3];
    expect(note?.kind === "note" && [note.label, inlineText(note.content)]).toEqual([
      "Note",
      "Dropout is applied according to dropout_p.",
    ]);
    expect(blocks[4]).toMatchObject({ kind: "note", label: "Added in 2.0" });
    expect(blocks[5]).toMatchObject({ kind: "paragraph" });
  });

  it("reads NumPy-style sections", () => {
    const blocks = formatDoc(
      [
        "Compute the exponential.",
        "",
        "Parameters",
        "----------",
        "x : array_like",
        "    Input values.",
        "out : ndarray, optional",
        "    A location into which the result is stored.",
        "",
        "Returns",
        "-------",
        "out : ndarray",
        "    Element-wise exponential of `x`.",
        "",
        "Examples",
        "--------",
        ">>> np.exp(1)",
        "2.718281828459045",
      ].join("\n"),
    );
    expect(kinds(blocks)).toEqual(["paragraph", "heading", "fields", "heading", "fields", "heading", "code"]);
    expect(blocks[2]).toMatchObject({
      kind: "fields",
      items: [
        { name: "x", type: "array_like" },
        { name: "out", type: "ndarray, optional" },
      ],
    });
  });

  it("reads lists, literal blocks after '::', and multi-line argument descriptions", () => {
    const blocks = formatDoc(
      [
        "Supports:",
        "",
        "- broadcasting to a common shape",
        "  across batch dimensions",
        "- type promotion",
        "",
        "Call it like this::",
        "",
        "    out = f(x)",
        "",
        "Args:",
        "    dim (int or tuple of ints): the dimension or dimensions",
        "        to reduce.",
        "    keepdim (bool): whether the output keeps `dim`.",
        "",
        "Trailing prose.",
      ].join("\n"),
    );
    expect(kinds(blocks)).toEqual([
      "paragraph",
      "list",
      "paragraph",
      "code",
      "heading",
      "fields",
      "paragraph",
    ]);
    const list = blocks[1];
    expect(list?.kind === "list" && list.items.map(inlineText)).toEqual([
      "broadcasting to a common shape across batch dimensions",
      "type promotion",
    ]);
    expect(blocks[2]).toEqual({ kind: "paragraph", content: [{ kind: "text", text: "Call it like this:" }] });
    const fields = blocks[5];
    expect(fields?.kind === "fields" && fields.items.map((field) => [field.name, field.type])).toEqual([
      ["dim", "int or tuple of ints"],
      ["keepdim", "bool"],
    ]);
    expect(fields?.kind === "fields" && inlineText(fields.items[0]?.content ?? [])).toBe(
      "the dimension or dimensions to reduce.",
    );
  });

  it("never produces markup: HTML in a docstring stays text", () => {
    const blocks = formatDoc('<img src=x onerror="alert(1)"> and <b>bold</b>');
    expect(blocks).toEqual([
      {
        kind: "paragraph",
        content: [{ kind: "text", text: '<img src=x onerror="alert(1)"> and <b>bold</b>' }],
      },
    ]);
    expect(formatDoc("")).toEqual([]);
  });
});

describe("parseInline", () => {
  it("turns roles, literals, and backticks into code, and math into math", () => {
    expect(parseInline("See :func:`~torch.nn.functional.softmax` and :class:`torch.Tensor`.")).toEqual([
      { kind: "text", text: "See " },
      { kind: "code", text: "softmax" },
      { kind: "text", text: " and " },
      { kind: "code", text: "torch.Tensor" },
      { kind: "text", text: "." },
    ]);
    expect(parseInline("Uses ``keepdim=True`` when :math:`n > 1`.")).toEqual([
      { kind: "text", text: "Uses " },
      { kind: "code", text: "keepdim=True" },
      { kind: "text", text: " when " },
      { kind: "math", text: "n > 1" },
      { kind: "text", text: "." },
    ]);
    // The docs helper has already turned `:math:` roles into single backticks.
    expect(parseInline("Letting `\\mathbb{K}` be `\\mathbb{R}`, with `k = min(m,n)`")).toEqual([
      { kind: "text", text: "Letting " },
      { kind: "math", text: "\\mathbb{K}" },
      { kind: "text", text: " be " },
      { kind: "math", text: "\\mathbb{R}" },
      { kind: "text", text: ", with " },
      { kind: "code", text: "k = min(m,n)" },
    ]);
  });

  it("reads emphasis and links, and unescapes Markdown", () => {
    expect(
      parseInline(
        "the **full SVD** of *any* matrix, see `the guide <https://x.test>`_ or [docs](https://y.test)",
      ),
    ).toEqual([
      { kind: "text", text: "the " },
      { kind: "strong", text: "full SVD" },
      { kind: "text", text: " of " },
      { kind: "emphasis", text: "any" },
      { kind: "text", text: " matrix, see the guide or docs" },
    ]);
    expect(parseInline("In-place version of exp\\_ with \\*args")).toEqual([
      { kind: "text", text: "In-place version of exp_ with *args" },
    ]);
    expect(parseInline("a * b and (input, *, out=None)")).toEqual([
      { kind: "text", text: "a * b and (input, *, out=None)" },
    ]);
  });
});

describe("looksLikeTex", () => {
  it("tells formulas from code", () => {
    expect(looksLikeTex("y_{i} = e^{x_{i}}")).toBe(true);
    expect(looksLikeTex("\\text{out} = \\frac{a}{b}")).toBe(true);
    expect(looksLikeTex(">>> x_{i} = 1")).toBe(false);
    expect(looksLikeTex('print("a\\nb")')).toBe(false);
    expect(looksLikeTex("k = min(m, n)")).toBe(false);
  });
});
