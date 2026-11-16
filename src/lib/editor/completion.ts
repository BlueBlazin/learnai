/** LSP `CompletionItemKind` values, as the word the list shows beside each name. */
const KIND_WORDS: Record<number, string> = {
  2: "method",
  3: "function",
  4: "class",
  5: "field",
  6: "variable",
  7: "class",
  8: "interface",
  9: "module",
  10: "property",
  12: "value",
  13: "enum",
  14: "keyword",
  20: "member",
  21: "constant",
  22: "class",
  25: "type",
};

/** The kind of a completion item in words ("function", "module"), or the empty string when unknown. */
export function kindWord(kind: number | undefined): string {
  return kind === undefined ? "" : (KIND_WORDS[kind] ?? "");
}

/**
 * The order of items that match what was typed equally well: public names
 * first, then `_private`, then `__dunder__`, each alphabetically. This
 * replaces the server's order, which lists modules and classes before
 * functions and so puts `export` above `exp` for `torch.ex`; with it, the name
 * that is a prefix of the others comes first.
 */
export function completionSortKey(label: string): string {
  const rank = label.startsWith("__") ? "2" : label.startsWith("_") ? "1" : "0";
  return `${rank}${label.toLowerCase()}`;
}

/** The text of an LSP documentation value (a string, or Markdown/plain-text markup content). */
export function markupText(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const parts = value.map(markupText).filter((part): part is string => part !== null);
    return parts.length ? parts.join("\n\n") : null;
  }
  if (value && typeof value === "object") {
    const content = value as { value?: unknown; language?: unknown };
    if (typeof content.value !== "string") return null;
    // A `MarkedString` with a language is code: fence it so the Markdown reader sees it that way.
    return typeof content.language === "string"
      ? `\`\`\`${content.language}\n${content.value}\n\`\`\``
      : content.value;
  }
  return null;
}

/** The most items the server sends in one list; a list this long was cut off. */
export const SERVER_LIST_LIMIT = 1000;

const WORD_ONLY = /^\w*$/;

/**
 * For how long a completion list stays good as the word being typed changes
 * (CodeMirror's `validFor`): while it holds, the list is narrowed in the
 * editor, at once; when it does not, the server is asked again.
 *
 * The server marks every list incomplete, because it sends only what matches
 * the letters typed so far. Such a list is still everything there is for any
 * longer word, so it stays good while the word grows, and is asked for again
 * when letters are deleted. Only a list that hit the server's limit may be
 * missing names for a longer word too.
 */
export function completionValidity(
  typed: string,
  incomplete: boolean,
  count: number,
): RegExp | ((text: string) => boolean) | undefined {
  if (!incomplete) return WORD_ONLY;
  if (count >= SERVER_LIST_LIMIT) return undefined;
  return (text) => text.startsWith(typed) && WORD_ONLY.test(text);
}
