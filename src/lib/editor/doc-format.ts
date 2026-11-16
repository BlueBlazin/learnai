/**
 * Documentation text, tidied into blocks the editor can lay out.
 *
 * Two dialects arrive here. The language server sends Markdown it converted
 * from docstrings (HTML entities, trailing-space line breaks, long fences,
 * leftover reStructuredText roles); the runtime docs helper sends the docstring
 * as plain text (Google- or NumPy-style sections, `.. math::` and other
 * directives, `Example::` literal blocks). Neither is parsed in full: the
 * common shapes become headings, argument lists, code, math, and notes, and
 * anything unrecognized stays a paragraph. The result is data, not HTML, so
 * nothing a docstring contains can reach the page as markup.
 */

export type DocInline = { kind: "text" | "code" | "strong" | "emphasis" | "math"; text: string };

export interface DocField {
  name: string;
  /** The declared type, when the docstring gives one. */
  type: string | null;
  content: DocInline[];
}

export type DocBlock =
  | { kind: "paragraph"; content: DocInline[] }
  | { kind: "heading"; text: string }
  | { kind: "code"; text: string }
  | { kind: "math"; text: string }
  | { kind: "list"; items: DocInline[][] }
  | { kind: "fields"; items: DocField[] }
  | { kind: "note"; label: string; content: DocInline[] };

/** What a hover from the language server holds: the code block on top, and the prose under it. */
export interface HoverParts {
  /** The signature or type, as code. Null when the server gave none. */
  signature: string | null;
  /** The documentation, still raw. Empty when the server has no docstring. */
  doc: string;
}

const ENTITIES: Record<string, string> = {
  "&lt;": "<",
  "&gt;": ">",
  "&amp;": "&",
  "&quot;": '"',
  "&#39;": "'",
  "&nbsp;": " ",
};

function normalize(text: string): string {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/&(?:lt|gt|amp|quot|#39|nbsp);/g, (entity) => ENTITIES[entity] ?? entity)
    .replace(/\t/g, "    ")
    .replace(/[ ]+$/gm, "");
}

const TEX_COMMAND =
  /\\(?:frac|text|mathbb|mathrm|mathbf|mathcal|operatorname|sum|prod|int|sqrt|left|right|begin|end|alpha|beta|gamma|delta|epsilon|sigma|lambda|mu|theta|pi|phi|omega|times|cdot|cdots|ldots|dots|leq|geq|neq|le|ge|ne|in|infty|exp|log|ln|max|min|quad|qquad|approx|sim|odot|otimes|oplus|top|star|lVert|rVert|Vert|vert|hat|bar|tilde|partial|nabla)(?![a-zA-Z])/;

/** True for text that is TeX rather than code: `\frac{a}{b}`, `x_{i}`, `e^{x}`. */
export function looksLikeTex(text: string): boolean {
  return (TEX_COMMAND.test(text) || /[_^]\{/.test(text)) && !/>>>|\bimport\b|\bdef\b/.test(text);
}

/**
 * Splits a Markdown hover into its leading code block and the rest. The
 * server separates the two with a rule (`---`).
 */
export function parseHover(markdown: string): HoverParts {
  const text = normalize(markdown).trim();
  const fence = /^(`{3,}|~{3,})[^\n]*\n([\s\S]*?)\n\1[ ]*(?:\n|$)/.exec(text);
  if (!fence) return { signature: null, doc: text };
  const rest = text
    .slice(fence[0].length)
    .replace(/^\s*-{3,}[ ]*(?:\n|$)/, "")
    .trim();
  const signature = (fence[2] ?? "").trim();
  return { signature: signature === "" ? null : signature, doc: rest };
}

/** True when the server could not say what a name is; such a hover is not worth showing. */
export function isUnknownSignature(signature: string | null): boolean {
  return signature === null || /^(Unknown|Never|Todo|@Todo)$/.test(signature.trim());
}

/** A signature this short is set on one line; a longer one keeps one parameter per line. */
const ONE_LINE_SIGNATURE = 72;

/**
 * A signature as the card shows it. `<module 'torch.nn'>` and `<class 'Tensor'>`
 * read better as `module torch.nn` and `class Tensor`, and the server writes
 * every signature with one parameter per line, which a short one does not need.
 */
export function plainSignature(signature: string): string {
  const described = /^<(module|class) '([^']+)'>$/.exec(signature.trim());
  if (described) return `${described[1]} ${described[2]}`;
  if (!signature.includes("\n")) return signature;
  const joined = signature
    .replace(/\(\n\s+/g, "(")
    .replace(/,\n\s+/g, ", ")
    .replace(/,?\n\)/g, ")");
  return joined.includes("\n") || joined.length > ONE_LINE_SIGNATURE ? signature : joined;
}

const SIGNATURE_LINE = /^[A-Za-z_][\w.]*\(.*\)(\s*->\s*.+)?$/;

/**
 * Drops a first paragraph that only repeats the call signature
 * (`exp(input, *, out=None) -> Tensor`): C functions carry it as the first
 * line of their docstring, and the card already shows the signature above.
 */
export function withoutLeadingSignature(text: string): string {
  const lines = normalize(text).split("\n");
  let index = 0;
  while (index < lines.length && (lines[index] ?? "").trim() === "") index += 1;
  const first = index;
  // The server's Markdown escapes underscores in names: `exp\_(input)`.
  const plain = (line: string) => line.trim().replace(/\\([_*`])/g, "$1");
  while (index < lines.length && SIGNATURE_LINE.test(plain(lines[index] ?? ""))) index += 1;
  if (index === first) return text;
  // Only a whole paragraph: a signature followed directly by prose is part of that prose.
  if (index < lines.length && (lines[index] ?? "").trim() !== "") return text;
  return lines.slice(index).join("\n");
}

const ROLE = /:(?:[a-z]+:)?([a-z-]+):`([^`]+)`/;
const INLINE = new RegExp(
  [
    ROLE.source, // 1 role name, 2 role text
    /``(.+?)``(?!`)/.source, // 3 literal
    /`([^`<]+?)\s*<[^`>]+>`_{1,2}/.source, // 4 link text (reStructuredText)
    /`([^`]+)`_{0,2}/.source, // 5 code or interpreted text
    /\*\*([^*\s](?:[^*]*[^*\s])?)\*\*/.source, // 6 strong
    /(?<![\w*\\])\*([^*\s](?:[^*]*[^*\s])?)\*(?![\w*])/.source, // 7 emphasis
    /\[([^\]]+)\]\([^)\s]+\)/.source, // 8 link text (Markdown)
    /\\([\\`*_{}[\]()#+\-.!<>|~])/.source, // 9 escaped character
  ].join("|"),
  "g",
);

/** The text a role shows: the last segment for `~torch.nn.Module`, the title of `title <target>`. */
function roleText(text: string): string {
  const titled = /^(.*?)\s*<[^>]+>$/.exec(text);
  const target = (titled?.[1] ?? text).trim();
  if (target.startsWith("~")) return target.slice(1).split(".").pop() ?? target.slice(1);
  return target.replace(/^!/, "");
}

/** Parses the inline markup of one paragraph: code, roles, math, strong, emphasis. */
export function parseInline(source: string): DocInline[] {
  const text = source.replace(/\s+/g, " ").trim();
  const out: DocInline[] = [];
  const push = (kind: DocInline["kind"], value: string) => {
    if (value === "") return;
    const last = out[out.length - 1];
    if (kind === "text" && last?.kind === "text") last.text += value;
    else out.push({ kind, text: value });
  };
  let cursor = 0;
  for (const match of text.matchAll(INLINE)) {
    push("text", text.slice(cursor, match.index));
    cursor = match.index + match[0].length;
    const [, role, roleBody, literal, rstLink, code, strong, emphasis, mdLink, escaped] = match;
    if (role !== undefined && roleBody !== undefined) {
      if (role === "math") push("math", roleBody);
      else push("code", roleText(roleBody));
    } else if (literal !== undefined) push("code", literal);
    else if (rstLink !== undefined) push("text", rstLink.trim());
    else if (code !== undefined) push(looksLikeTex(code) ? "math" : "code", code);
    else if (strong !== undefined) push("strong", strong);
    else if (emphasis !== undefined) push("emphasis", emphasis);
    else if (mdLink !== undefined) push("text", mdLink);
    else if (escaped !== undefined) push("text", escaped);
  }
  push("text", text.slice(cursor));
  return out;
}

const FIELD_SECTIONS = new Set([
  "args",
  "arguments",
  "parameters",
  "params",
  "keyword args",
  "keyword arguments",
  "other parameters",
  "attributes",
  "raises",
  "yields",
  "returns",
  "return",
]);
const SECTIONS = new Set([
  ...FIELD_SECTIONS,
  "example",
  "examples",
  "note",
  "notes",
  "shape",
  "see also",
  "references",
  "warning",
  "warnings",
]);
const HEADING_TEXT: Record<string, string> = {
  args: "Arguments",
  params: "Parameters",
  "keyword args": "Keyword arguments",
  "keyword arguments": "Keyword arguments",
  return: "Returns",
  example: "Example",
  examples: "Examples",
};
const CODE_DIRECTIVES = new Set([
  "code-block",
  "code",
  "sourcecode",
  "testcode",
  "doctest",
  "parsed-literal",
]);
const NOTE_DIRECTIVES: Record<string, string> = {
  note: "Note",
  warning: "Warning",
  tip: "Tip",
  hint: "Tip",
  important: "Important",
  caution: "Caution",
  attention: "Attention",
  seealso: "See also",
  deprecated: "Deprecated",
  versionadded: "Added in",
  versionchanged: "Changed in",
};

const FENCE = /^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$/;
const DIRECTIVE = /^(\s*)\.\. ([\w-]+)::\s*(.*)$/;
const BULLET = /^(\s*)[-*+]\s+(.*)$/;
const MARKDOWN_HEADING = /^\s*#{1,6}\s+(.*?)\s*#*$/;
const MARKDOWN_FIELD = /^\*\*([^*]+)\*\*:\s*(?:`([^`]*)`)?\s*(.*)$/;
const GOOGLE_FIELD = /^(\*{0,2}[\w.]+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$/;
const NUMPY_FIELD = /^(\*{0,2}[\w.]+(?:\s*,\s*\*{0,2}[\w.]+)*)\s+:\s*(.*)$/;
const UNDERLINE = /^\s*([-=~^"'`#*+])\1{2,}\s*$/;

function indentOf(line: string): number {
  return line.length - line.trimStart().length;
}

function dedent(lines: string[]): string {
  const filled = lines.filter((line) => line.trim() !== "");
  const margin = filled.length ? Math.min(...filled.map(indentOf)) : 0;
  const out = lines.map((line) => line.slice(margin));
  while (out.length && (out[0] ?? "").trim() === "") out.shift();
  while (out.length && (out[out.length - 1] ?? "").trim() === "") out.pop();
  return out.join("\n");
}

function headingText(raw: string): string {
  const text = raw.replace(/:+$/, "").trim();
  return HEADING_TEXT[text.toLowerCase()] ?? text;
}

/** Turns documentation text (either dialect) into blocks. Never throws; unknown markup stays text. */
export function formatDoc(source: string): DocBlock[] {
  const lines = normalize(source).split("\n");
  const blocks: DocBlock[] = [];
  /** The indent of the section heading whose entries are fields, or null outside such a section. */
  let fieldSection: number | null = null;
  let i = 0;

  const line = (index: number) => lines[index] ?? "";
  const blank = (index: number) => line(index).trim() === "";

  /** Lines indented deeper than `base`, starting at `from` (blank lines before and inside included). */
  const takeIndented = (from: number, base: number): { text: string; next: number } => {
    let end = from;
    let last = from;
    while (end < lines.length && (blank(end) || indentOf(line(end)) > base)) {
      if (!blank(end)) last = end + 1;
      end += 1;
    }
    return { text: dedent(lines.slice(from, last)), next: last };
  };

  const pushCode = (text: string, language = "") => {
    if (text.trim() === "") return;
    const math = language === "" || language === "python" ? looksLikeTex(text) : false;
    blocks.push({ kind: math ? "math" : "code", text });
  };

  const pushField = (name: string, type: string | null, description: string) => {
    const field: DocField = {
      name,
      type: type?.trim() || null,
      content: parseInline(description),
    };
    const last = blocks[blocks.length - 1];
    if (last?.kind === "fields") last.items.push(field);
    else blocks.push({ kind: "fields", items: [field] });
  };

  const startsBlock = (index: number): boolean => {
    const text = line(index);
    return (
      FENCE.test(text) ||
      DIRECTIVE.test(text) ||
      BULLET.test(text) ||
      MARKDOWN_HEADING.test(text) ||
      MARKDOWN_FIELD.test(text.trim()) ||
      text.trimStart().startsWith(">>>")
    );
  };

  while (i < lines.length) {
    if (blank(i)) {
      i += 1;
      continue;
    }
    const text = line(i);
    const indent = indentOf(text);
    const trimmed = text.trim();
    if (fieldSection !== null && indent <= fieldSection && !MARKDOWN_FIELD.test(trimmed)) {
      fieldSection = null;
    }

    const fence = FENCE.exec(text);
    if (fence) {
      const marker = fence[1] ?? "```";
      let end = i + 1;
      while (end < lines.length && !line(end).trim().startsWith(marker)) end += 1;
      pushCode(dedent(lines.slice(i + 1, end)), fence[2]);
      i = end + 1;
      continue;
    }

    const directive = DIRECTIVE.exec(text);
    if (directive) {
      const name = (directive[2] ?? "").toLowerCase();
      const argument = (directive[3] ?? "").trim();
      const body = takeIndented(i + 1, indent);
      i = body.next;
      if (name === "math") {
        const tex = [argument, body.text].filter(Boolean).join("\n");
        if (tex) blocks.push({ kind: "math", text: tex });
      } else if (CODE_DIRECTIVES.has(name)) {
        if (body.text) blocks.push({ kind: "code", text: body.text });
      } else if (name in NOTE_DIRECTIVES) {
        const label = NOTE_DIRECTIVES[name] ?? "Note";
        const versioned = name.startsWith("version") || name === "deprecated";
        blocks.push({
          kind: "note",
          label:
            versioned && argument ? `${label} ${argument}`.replace("Deprecated ", "Deprecated in ") : label,
          content: parseInline(versioned ? body.text : [argument, body.text].filter(Boolean).join(" ")),
        });
      } else {
        // An unknown directive (`.. function::`, `.. autosummary::`): keep what it says, drop the
        // markup and its options (`:noindex:`).
        const said = body.text
          .split("\n")
          .filter((entry) => !/^\s*:[\w-]+:/.test(entry))
          .join("\n");
        if (said.trim()) blocks.push(...formatDoc(said));
      }
      continue;
    }
    // A comment or a link target (`.. _label:`): nothing to show.
    if (/^\s*\.\.(\s|$)/.test(text)) {
      i = takeIndented(i + 1, indent).next;
      continue;
    }

    const heading = MARKDOWN_HEADING.exec(text);
    if (heading) {
      const title = headingText(heading[1] ?? "");
      blocks.push({ kind: "heading", text: title });
      fieldSection = FIELD_SECTIONS.has((heading[1] ?? "").replace(/:+$/, "").trim().toLowerCase())
        ? indent - 1
        : null;
      i += 1;
      continue;
    }
    // NumPy style: a title with a rule of dashes under it.
    if (UNDERLINE.test(line(i + 1)) && trimmed.length <= 40 && !UNDERLINE.test(text)) {
      blocks.push({ kind: "heading", text: headingText(trimmed) });
      fieldSection = FIELD_SECTIONS.has(trimmed.toLowerCase()) ? indent - 1 : null;
      i += 2;
      continue;
    }
    if (UNDERLINE.test(text)) {
      i += 1;
      continue;
    }
    // Google style: `Args:` alone on its line. `Example::` also opens a literal block.
    const section = /^([A-Za-z][A-Za-z ]{0,30})(::?)$/.exec(trimmed);
    if (section && SECTIONS.has((section[1] ?? "").toLowerCase())) {
      blocks.push({ kind: "heading", text: headingText(section[1] ?? "") });
      fieldSection = FIELD_SECTIONS.has((section[1] ?? "").toLowerCase()) ? indent : null;
      i += 1;
      if (section[2] === "::") {
        const body = takeIndented(i, indent);
        pushCode(body.text);
        i = body.next;
      }
      continue;
    }

    if (trimmed.startsWith(">>>")) {
      let end = i;
      while (end < lines.length && !blank(end)) end += 1;
      blocks.push({ kind: "code", text: dedent(lines.slice(i, end)) });
      i = end;
      continue;
    }

    const bullet = BULLET.exec(text);
    if (bullet) {
      const items: DocInline[][] = [];
      while (i < lines.length) {
        const item = BULLET.exec(line(i));
        if (!item || indentOf(line(i)) !== indent) break;
        const parts = [item[2] ?? ""];
        i += 1;
        while (i < lines.length && !blank(i) && indentOf(line(i)) > indent && !BULLET.test(line(i))) {
          parts.push(line(i).trim());
          i += 1;
        }
        items.push(parseInline(parts.join(" ")));
        // A blank line between items does not end the list.
        if (blank(i) && BULLET.test(line(i + 1)) && indentOf(line(i + 1)) === indent) i += 1;
      }
      blocks.push({ kind: "list", items });
      continue;
    }

    // The server's Markdown for an argument: `**input**: `Tensor`` and the description under it.
    const markdownField = MARKDOWN_FIELD.exec(trimmed);
    if (markdownField) {
      const parts = [markdownField[3] ?? ""];
      i += 1;
      while (i < lines.length && !blank(i) && !startsBlock(i)) {
        parts.push(line(i).trim());
        i += 1;
      }
      pushField(markdownField[1] ?? "", markdownField[2] ?? null, parts.join(" "));
      continue;
    }

    if (fieldSection !== null) {
      // NumPy writes `name : type` with the description indented below; Google `name (type): description`.
      const below = indentOf(line(i + 1)) > indent && !blank(i + 1);
      const numpy = below ? NUMPY_FIELD.exec(trimmed) : null;
      const google = numpy ? null : GOOGLE_FIELD.exec(trimmed);
      // NumPy also writes a bare name (or a bare type under Returns) with the description indented below.
      const bare = !google && !numpy && below && /^[\w.*]+$/.test(trimmed);
      if (google || numpy || bare) {
        const parts = [google ? (google[3] ?? "") : ""];
        i += 1;
        while (i < lines.length && !blank(i) && indentOf(line(i)) > indent) {
          parts.push(line(i).trim());
          i += 1;
        }
        const name = google ? (google[1] ?? "") : numpy ? (numpy[1] ?? "") : trimmed;
        const type = google ? (google[2] ?? null) : numpy ? (numpy[2] ?? null) : null;
        pushField(name, type, parts.join(" "));
        continue;
      }
    }

    // A paragraph: up to a blank line, a change to deeper structure, or the start of another block.
    const parts: string[] = [];
    while (i < lines.length && !blank(i) && (parts.length === 0 || !startsBlock(i))) {
      if (parts.length > 0 && UNDERLINE.test(line(i + 1))) break;
      parts.push(line(i).trim());
      i += 1;
    }
    let paragraph = parts.join(" ");
    // `text::` announces a literal block: the indented lines that follow are code.
    if (paragraph.endsWith("::")) {
      paragraph = paragraph.slice(0, -2).trimEnd();
      if (paragraph !== "" && !/[.:!?]$/.test(paragraph)) paragraph += ":";
      const body = takeIndented(i, indent);
      const content = parseInline(paragraph);
      if (content.length) blocks.push({ kind: "paragraph", content });
      pushCode(body.text);
      i = body.next;
      continue;
    }
    const content = parseInline(paragraph);
    if (content.length) blocks.push({ kind: "paragraph", content });
  }
  return blocks;
}

/** The plain text of inline content, for a one-line summary or an accessible label. */
export function inlineText(content: readonly DocInline[]): string {
  return content.map((part) => part.text).join("");
}
