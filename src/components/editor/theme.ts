import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import type { Extension } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { tags as t } from "@lezer/highlight";

/**
 * The editor theme. Every value is a CSS variable from styles/tokens.css, so the
 * same extension is correct in light and dark with no reconfiguration.
 */
const chrome = EditorView.theme({
  "&": {
    color: "var(--foreground)",
    backgroundColor: "transparent",
    fontSize: "var(--text-code)",
  },
  ".cm-scroller": {
    fontFamily: "var(--font-mono)",
    lineHeight: "var(--text-code--line-height)",
    fontVariantLigatures: "none",
    overflow: "auto",
  },
  ".cm-content": {
    padding: "0.75rem 0 4rem",
    caretColor: "var(--primary)",
  },
  ".cm-line": { padding: "0 1rem 0 0.25rem" },
  ".cm-gutters": {
    backgroundColor: "var(--paper)",
    color: "var(--faint)",
    border: "none",
    paddingLeft: "0.5rem",
  },
  ".cm-lineNumbers .cm-gutterElement": {
    padding: "0 0.75rem 0 0.5rem",
    minWidth: "2rem",
    fontVariantNumeric: "tabular-nums",
  },
  ".cm-activeLine": { backgroundColor: "var(--editor-active-line)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--foreground)" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--primary)", borderLeftWidth: "1.5px" },
  "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground": {
    backgroundColor: "var(--editor-selection)",
  },
  ".cm-content ::selection": { backgroundColor: "var(--editor-selection)" },
  "&.cm-focused .cm-matchingBracket": {
    backgroundColor: "var(--straw)",
    color: "inherit",
    outline: "none",
  },
  "&.cm-focused .cm-nonmatchingBracket": { backgroundColor: "var(--fail-subtle)" },
  ".cm-selectionMatch": { backgroundColor: "var(--accent)" },
  ".cm-searchMatch": { backgroundColor: "var(--straw)", outline: "none" },
  ".cm-searchMatch.cm-searchMatch-selected": { outline: "1.5px solid var(--straw-foreground)" },
  ".cm-foldPlaceholder": {
    backgroundColor: "var(--muted)",
    border: "none",
    color: "var(--muted-foreground)",
  },
  ".cm-tooltip": {
    backgroundColor: "var(--popover)",
    color: "var(--popover-foreground)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-lg)",
    boxShadow: "var(--shadow-float)",
    overflow: "hidden",
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    maxHeight: "14rem",
  },
  ".cm-tooltip-autocomplete ul li": { padding: "0.2rem 0.6rem" },
  ".cm-tooltip-autocomplete ul li[aria-selected]": {
    backgroundColor: "var(--accent)",
    color: "var(--accent-foreground)",
  },
  ".cm-completionIcon": { display: "none" },
  ".cm-panels": {
    backgroundColor: "var(--background)",
    color: "var(--foreground)",
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-xs)",
  },
  ".cm-panels.cm-panels-top": { borderBottom: "1px solid var(--border)" },
  ".cm-panels.cm-panels-bottom": { borderTop: "1px solid var(--border)" },
  ".cm-panel.cm-search": { padding: "0.5rem 0.75rem" },
  ".cm-panel.cm-search label": { color: "var(--muted-foreground)", fontSize: "var(--text-xs)" },
  ".cm-textfield": {
    backgroundColor: "var(--paper)",
    border: "1px solid var(--input)",
    borderRadius: "var(--radius-md)",
    color: "var(--foreground)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    padding: "0.2rem 0.4rem",
  },
  ".cm-button": {
    backgroundImage: "none",
    backgroundColor: "transparent",
    border: "1px solid var(--input)",
    borderRadius: "var(--radius-md)",
    color: "var(--foreground)",
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-xs)",
    padding: "0.2rem 0.5rem",
  },
  ".cm-button:active": { backgroundImage: "none", backgroundColor: "var(--accent)" },
  ".cm-panel.cm-search [name=close]": { color: "var(--muted-foreground)", fontSize: "var(--text-md)" },
});

/** Four inks: keywords, strings, numbers and constants, comments. Everything else is body ink. */
const syntax = HighlightStyle.define([
  {
    tag: [t.keyword, t.controlKeyword, t.operatorKeyword, t.definitionKeyword, t.moduleKeyword],
    color: "var(--syntax-keyword)",
  },
  { tag: [t.string, t.special(t.string), t.docString], color: "var(--syntax-string)" },
  { tag: [t.number, t.bool, t.null, t.atom, t.self], color: "var(--syntax-number)" },
  { tag: [t.comment, t.lineComment, t.blockComment], color: "var(--syntax-comment)", fontStyle: "italic" },
  {
    tag: [t.function(t.definition(t.variableName)), t.definition(t.className)],
    color: "var(--syntax-function)",
    fontWeight: "600",
  },
  { tag: [t.punctuation, t.bracket, t.operator], color: "var(--syntax-punctuation)" },
  { tag: t.meta, color: "var(--syntax-comment)" },
  { tag: t.invalid, color: "var(--fail)" },
]);

export const learnaiEditorTheme: Extension = [chrome, syntaxHighlighting(syntax)];
