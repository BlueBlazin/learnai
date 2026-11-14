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
 * A paragraph that holds nothing but an image becomes a numbered figure:
 *
 *     ![The caption, set beneath the figure.](assets/rotation.svg)
 *
 * The paragraph becomes a `<figure data-src data-number data-alt data-caption>`;
 * figures are numbered in document order from 1. `data-caption` is the caption
 * as the author wrote it (Markdown: it may hold math and emphasis), `data-alt`
 * the same as plain text. An image inside a sentence stays an inline image.
 *
 * And punctuation that follows inline math is bound to it (`bindMathPunctuation`).
 */

/** Image URLs under this prefix name a file in the problem's `assets/` directory. */
export const ASSET_PREFIX = "assets/";

interface Node {
  type: string;
  value?: string;
  url?: string;
  alt?: string | null;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: Node[];
  data?: { hName?: string; hProperties?: Record<string, unknown> };
}

const MARKER = /^\[!(NOTE|TIP|WARNING)\][ \t]*\n?/i;

/** The image of a paragraph that contains only that image (and whitespace), if it is one. */
function soleImage(node: Node): Node | null {
  if (node.type !== "paragraph") return null;
  const content = (node.children ?? []).filter((child) => !(child.type === "text" && !child.value?.trim()));
  const only = content.length === 1 ? content[0] : undefined;
  return only?.type === "image" ? only : null;
}

/**
 * The caption of an image as written between `![` and `](`. The syntax tree
 * keeps only the plain-text `alt`, which loses math and emphasis, so the label
 * is cut from the source; without positions the plain text stands in.
 */
function captionSource(image: Node, source: string): string {
  const start = image.position?.start.offset;
  const end = image.position?.end.offset;
  const raw = start !== undefined && end !== undefined ? source.slice(start, end) : "";
  const close = raw.lastIndexOf("](");
  return raw.startsWith("![") && close !== -1 ? raw.slice(2, close) : (image.alt ?? "");
}

interface State {
  source: string;
  figures: number;
}

function visit(node: Node, counter: State): void {
  const image = soleImage(node);
  if (image) {
    counter.figures += 1;
    node.children = [];
    node.data = {
      ...node.data,
      hName: "figure",
      hProperties: {
        "data-src": image.url ?? "",
        "data-number": counter.figures,
        "data-alt": image.alt ?? "",
        "data-caption": captionSource(image, counter.source),
      },
    };
    return;
  }
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
  for (const child of node.children ?? []) visit(child, counter);
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
  return (tree: unknown, file?: unknown) =>
    visit(tree as Node, { source: file === undefined ? "" : String(file), figures: 0 });
}
