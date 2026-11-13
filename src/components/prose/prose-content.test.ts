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

  it("renders the same document repeatedly without growing the tree", () => {
    const first = render(SAMPLE);
    for (let i = 0; i < 5; i++) expect(render(SAMPLE)).toBe(first);
  });
});
