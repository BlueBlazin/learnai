import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ProseContent } from "./prose-content";

const render = (markdown: string) => renderToStaticMarkup(createElement(ProseContent, null, markdown));

const SAMPLE = `
Implement \`softmax\` for a vector $x \\in \\mathbb{R}^n$:

$$\\operatorname{softmax}(x)_i = \\frac{e^{x_i}}{\\sum_{j=1}^{n} e^{x_j}}.$$

- Shift by $c$, then by $m = \\max_j x_j$; the largest term is $e^0 = 1$.
  - Nested: for $d_v$, it computes $x$ (and $y$), then $z$.

> [!WARNING]
> With $x_i = 1000$, \`exp\` overflows.

| input | output |
|---|---|
| $[1, 2, 3]$, | $0.09$. |

\`\`\`python
def softmax(x, axis=-1): ...
\`\`\`
`;

describe("ProseContent", () => {
  it("renders math, callouts, tables, and code, binding punctuation to inline math", () => {
    const html = render(SAMPLE);
    expect(html).toContain('class="katex"');
    expect(html).not.toContain("katex-error");
    expect(html).toContain('data-kind="warning"');
    expect(html).toContain("<table>");
    expect(html).toContain("code-block");
    expect(html.match(/math-nowrap/g)?.length).toBeGreaterThanOrEqual(5);
  });

  it("turns an image on its own line into a numbered figure with a typeset caption", () => {
    const html = render(
      [
        "Before.",
        "![The pair $(x_0, x_1)$ turns by *one* radian.](assets/rotation.svg)",
        "An inline ![mark](assets/mark.png) stays inline.",
        "![Second figure](https://example.com/plot.png)",
      ].join("\n\n"),
    );
    expect(html.match(/<figure class="figure">/g)?.length).toBe(2);
    // Numbered in document order; the caption keeps its math and emphasis.
    expect(html).toMatch(/Figure 1\.<\/span> The pair <span class="katex">.*<em>one<\/em> radian\./);
    expect(html).toContain("Figure 2.</span> Second figure");
    // Without a problem to resolve against, an asset is named, not requested.
    expect(html).toContain("This figure belongs to a problem");
    expect(html).not.toContain('src="assets/rotation.svg"');
    // The plain-text alternative has no markup in it.
    expect(html).toContain('<img src="https://example.com/plot.png" alt="Second figure"/>');
    expect(html).toContain('<p>An inline <img src="assets/mark.png" alt="mark"/> stays inline.</p>');
  });

  it("draws interactive figures from the registry, numbered with the others", () => {
    const html = render(
      [
        "![Ten tokens under $k$.](interactive:sampler-order)",
        "![Never built.](interactive:no-such-figure)",
        "![A plot](https://example.com/plot.png)",
      ].join("\n\n"),
    );
    expect(html.match(/<figure class="figure">/g)?.length).toBe(3);
    expect(html).toMatch(/Figure 1\.<\/span> Ten tokens under <span class="katex">/);
    expect(html).toContain("There is no interactive figure named <code>no-such-figure</code>.");
    expect(html).toContain("Figure 2.</span> Never built.");
    expect(html).toContain("Figure 3.</span> A plot");
    // The name is never treated as a URL to load.
    expect(html).not.toContain('src="interactive:');
  });

  it("renders the same document repeatedly without growing the tree", () => {
    const first = render(SAMPLE);
    for (let i = 0; i < 5; i++) expect(render(SAMPLE)).toBe(first);
  });
});
