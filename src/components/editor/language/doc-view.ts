import { pythonLanguage } from "@codemirror/lang-python";
import { highlightingFor } from "@codemirror/language";
import type { EditorState } from "@codemirror/state";
import type { EditorView } from "@codemirror/view";
import { highlightCode } from "@lezer/highlight";
import type { DocCardContent } from "@/lib/editor/doc-card";
import type { DocBlock, DocInline } from "@/lib/editor/doc-format";

/**
 * The documentation card shared by hover, the completion side panel, and
 * signature help: a signature in the code face over tidied prose. Everything
 * is built with `createElement` and `textContent`; documentation is never
 * parsed as HTML. Styling lives in `../language-theme.ts`.
 */

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

type Katex = typeof import("katex").default;
let katexModule: Promise<Katex> | null = null;

/**
 * Typesets TeX in place once KaTeX (already part of the app for statements)
 * has loaded. Until then, and whenever the source is not valid TeX, the node
 * keeps the source text in the code face.
 */
function typeset(node: HTMLElement, tex: string, display: boolean, view: EditorView): void {
  node.textContent = tex;
  katexModule ??= import("katex").then((module) => module.default);
  void katexModule
    .then((katex) => {
      const rendered = document.createElement("span");
      katex.render(tex, rendered, { displayMode: display, throwOnError: true });
      node.replaceChildren(rendered);
      node.dataset.typeset = "true";
      // The card changed size: let its tooltip find its place again.
      view.requestMeasure();
    })
    .catch(() => {});
}

function inline(parent: HTMLElement, content: readonly DocInline[], view: EditorView): void {
  for (const part of content) {
    if (part.kind === "text") parent.append(part.text);
    else if (part.kind === "code") parent.append(element("code", "cm-doc-code-inline", part.text));
    else if (part.kind === "strong") parent.append(element("strong", "cm-doc-strong", part.text));
    else if (part.kind === "emphasis") parent.append(element("em", "cm-doc-emphasis", part.text));
    else {
      const math = element("span", "cm-doc-math-inline");
      typeset(math, part.text, false, view);
      parent.append(math);
    }
  }
}

function block(item: DocBlock, view: EditorView): HTMLElement {
  switch (item.kind) {
    case "heading":
      return element("div", "cm-doc-heading", item.text);
    case "code":
      return element("pre", "cm-doc-code", item.text);
    case "math": {
      const math = element("div", "cm-doc-math");
      typeset(math, item.text, true, view);
      return math;
    }
    case "list": {
      const list = element("ul", "cm-doc-list");
      for (const entry of item.items) {
        const row = element("li", "cm-doc-list-item");
        inline(row, entry, view);
        list.append(row);
      }
      return list;
    }
    case "fields": {
      const list = element("dl", "cm-doc-fields");
      for (const field of item.items) {
        const term = element("dt", "cm-doc-field-name");
        term.append(element("code", "cm-doc-code-inline", field.name));
        if (field.type) term.append(element("span", "cm-doc-field-type", field.type));
        const description = element("dd", "cm-doc-field-text");
        inline(description, field.content, view);
        list.append(term, description);
      }
      return list;
    }
    case "note": {
      const note = element("div", "cm-doc-note");
      note.append(element("span", "cm-doc-note-label", item.label));
      const text = element("span", "cm-doc-note-text");
      inline(text, item.content, view);
      note.append(text);
      return note;
    }
    default: {
      const paragraph = element("p", "cm-doc-paragraph");
      inline(paragraph, item.content, view);
      return paragraph;
    }
  }
}

/** Code with the editor's own syntax colors. Python only; anything else is set plain. */
export function highlighted(state: EditorState, code: string, className: string): HTMLElement {
  const node = element("div", className);
  if (!/^(def|class|async)\b|[(:=]/.test(code)) {
    node.textContent = code;
    return node;
  }
  try {
    highlightCode(
      code,
      pythonLanguage.parser.parse(code),
      { style: (tags) => highlightingFor(state, tags) },
      (text, classes) => {
        if (!classes) node.append(text);
        else node.append(element("span", classes, text));
      },
      () => node.append("\n"),
    );
  } catch {
    node.textContent = code;
  }
  return node;
}

/** Fills (or refills) a card. Refilling is how a late answer from the docs helper lands in an open card. */
export function fillDocCard(card: HTMLElement, view: EditorView, content: DocCardContent): void {
  const children: HTMLElement[] = [];
  if (content.signature) children.push(highlighted(view.state, content.signature, "cm-doc-signature"));
  if (content.caption) children.push(element("div", "cm-doc-caption", content.caption));
  if (content.blocks?.length) {
    const body = element("div", "cm-doc-body");
    for (const item of content.blocks) body.append(block(item, view));
    if (content.truncated)
      body.append(element("p", "cm-doc-truncated", "The rest is in the library's documentation."));
    children.push(body);
  }
  card.replaceChildren(...children);
}

export function docCard(view: EditorView, content: DocCardContent, className = ""): HTMLElement {
  const card = element("div", `cm-doc-card ${className}`.trim());
  fillDocCard(card, view, content);
  return card;
}
