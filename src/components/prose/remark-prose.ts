/**
 * The Markdown transforms prose needs beyond GFM and math, as one remark plugin.
 *
 * GitHub-style alerts become callouts:
 *
 *     > [!NOTE]
 *     > Text of the note.
 *
 * The blockquote becomes an `<aside class="callout" data-kind="note">`. Kinds:
 * note, tip, warning. Anything else stays an ordinary quotation.
 *
 * And punctuation that follows inline math is bound to it (`bindMathPunctuation`).
 */

interface Node {
  type: string;
  value?: string;
  children?: Node[];
  data?: { hName?: string; hProperties?: Record<string, unknown> };
}

const MARKER = /^\[!(NOTE|TIP|WARNING)\][ \t]*\n?/i;

function visit(node: Node): void {
  if (node.type === "blockquote") {
    const first = node.children?.[0];
    const text = first?.type === "paragraph" ? first.children?.[0] : undefined;
    const match = text?.type === "text" && text.value ? MARKER.exec(text.value) : null;
    if (match && text?.value !== undefined && first?.children) {
      const kind = (match[1] ?? "note").toLowerCase();
      text.value = text.value.slice(match[0].length);
      if (text.value === "") first.children.shift();
      node.data = {
        ...node.data,
        hName: "aside",
        hProperties: { className: ["callout"], "data-kind": kind },
      };
    }
  }
  if (node.children && node.type !== "mathWithPunctuation") {
    node.children = bindMathPunctuation(node.children);
  }
  for (const child of node.children ?? []) visit(child);
}

const LEADING_PUNCTUATION = /^[,.;:!?)\]]+/;

/**
 * Keeps punctuation that follows inline math on the same line as the math.
 * Without this the browser may wrap between a formula and its comma.
 */
function bindMathPunctuation(children: Node[]): Node[] {
  const out: Node[] = [];
  for (let i = 0; i < children.length; i++) {
    const node = children[i] as Node;
    const next = children[i + 1];
    const match =
      node.type === "inlineMath" && next?.type === "text" && next.value
        ? LEADING_PUNCTUATION.exec(next.value)
        : null;
    if (match && next?.value !== undefined) {
      next.value = next.value.slice(match[0].length);
      out.push({
        type: "mathWithPunctuation",
        data: { hName: "span", hProperties: { className: ["math-nowrap"] } },
        children: [node, { type: "text", value: match[0] }],
      });
    } else {
      out.push(node);
    }
  }
  return out;
}

export function remarkProse() {
  return (tree: unknown) => visit(tree as Node);
}
