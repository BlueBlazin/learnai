import { EditorView } from "@codemirror/view";

/**
 * The look of what the language server adds: the completion list and its
 * documentation panel, the hover card, signature help, and diagnostics. Like
 * the rest of the editor theme, every value is a token from styles/tokens.css,
 * so one definition is right in light and dark. The floating layers follow the
 * app's popover (surface, hairline, the one elevation); rows follow the
 * command palette; documentation is set in the interface face with code in
 * the mono face. `language/doc-view.ts` builds the card these rules style.
 */
const hairline = "color-mix(in oklch, var(--foreground) 12%, transparent)";

export const languageTheme = EditorView.theme({
  // Floating layers ---------------------------------------------------------
  ".cm-tooltip": {
    backgroundColor: "var(--popover)",
    color: "var(--popover-foreground)",
    border: `1px solid ${hairline}`,
    borderRadius: "var(--radius-lg)",
    boxShadow: "var(--shadow-float)",
    overflow: "hidden",
  },
  ".cm-tooltip-section:not(:first-child)": { borderTop: "1px solid var(--border)" },

  // Completion list: rows like the command palette, the kind as a word on the right.
  // The documentation panel is a child positioned beside the list, so the list's own box must not clip.
  ".cm-tooltip.cm-tooltip-autocomplete": { overflow: "visible" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    lineHeight: "1.25rem",
    minWidth: "17rem",
    maxWidth: "30rem",
    maxHeight: "15.5rem",
    padding: "0.25rem",
    scrollPadding: "0.25rem",
  },
  ".cm-tooltip-autocomplete ul li": {
    display: "flex",
    alignItems: "baseline",
    gap: "1.5rem",
    padding: "0.0625rem 0.5rem",
    borderRadius: "var(--radius-md)",
  },
  ".cm-tooltip-autocomplete ul li[aria-selected]": {
    backgroundColor: "var(--accent)",
    color: "var(--accent-foreground)",
  },
  ".cm-completionIcon": { display: "none" },
  ".cm-completionLabel": {
    flex: "1 1 auto",
    minWidth: "0",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  ".cm-completionMatchedText": {
    textDecoration: "none",
    fontWeight: "600",
    color: "var(--primary)",
  },
  ".cm-completionDetail": {
    flex: "none",
    marginLeft: "auto",
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-2xs)",
    fontStyle: "normal",
    color: "var(--muted-foreground)",
  },
  ".cm-tooltip.cm-completionInfo": {
    padding: "0",
    width: "26rem",
    maxWidth: "min(26rem, 45vw)",
  },
  ".cm-completionInfo.cm-completionInfo-right": { marginLeft: "0.375rem" },
  ".cm-completionInfo.cm-completionInfo-left": { marginRight: "0.375rem" },

  // The documentation card ----------------------------------------------------
  // A card never outgrows the room beside its anchor: the signature and the prose scroll inside it.
  ".cm-doc-card": {
    display: "flex",
    flexDirection: "column",
    maxHeight: "min(22rem, 44vh)",
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-sm)",
    lineHeight: "var(--text-sm--line-height)",
    color: "var(--popover-foreground)",
    cursor: "text",
    userSelect: "text",
    WebkitUserSelect: "text",
  },
  ".cm-doc-card > *": { flex: "none" },
  ".cm-doc-card-hover": { width: "max-content", maxWidth: "min(34rem, 80vw)" },
  ".cm-doc-card-signature": { width: "max-content", maxWidth: "min(38rem, 80vw)" },
  ".cm-doc-signature": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    lineHeight: "1.1875rem",
    whiteSpace: "pre-wrap",
    overflowWrap: "anywhere",
    padding: "0.5rem 0.75rem",
    backgroundColor: "var(--code)",
    maxHeight: "min(10.5rem, 22vh)",
    overflowY: "auto",
  },
  ".cm-doc-signature:not(:last-child)": { borderBottom: "1px solid var(--border)" },
  ".cm-doc-active-parameter": {
    backgroundColor: "var(--straw)",
    borderRadius: "var(--radius-sm)",
    fontWeight: "600",
    boxShadow: "0 0 0 2px var(--straw)",
  },
  ".cm-doc-parameter": {
    padding: "0.375rem 0.75rem",
    fontSize: "var(--text-xs)",
    lineHeight: "1.125rem",
    color: "var(--muted-foreground)",
  },
  ".cm-doc-caption": {
    padding: "0.4375rem 0.75rem 0",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-2xs)",
    lineHeight: "var(--text-2xs--line-height)",
    color: "var(--muted-foreground)",
    overflowWrap: "anywhere",
  },
  ".cm-doc-caption:last-child": { paddingBottom: "0.4375rem" },
  ".cm-doc-card > .cm-doc-body": { flex: "1 1 auto" },
  ".cm-doc-body": {
    padding: "0.5rem 0.75rem 0.625rem",
    minHeight: "0",
    overflowY: "auto",
    overscrollBehavior: "contain",
    textWrap: "pretty",
    overflowWrap: "break-word",
  },
  ".cm-doc-body > *": { margin: "0" },
  ".cm-doc-body > * + *": { marginTop: "0.5rem" },
  ".cm-doc-heading": {
    fontSize: "var(--text-xs)",
    lineHeight: "var(--text-xs--line-height)",
    fontWeight: "600",
    color: "var(--foreground)",
  },
  ".cm-doc-body > * + .cm-doc-heading": {
    marginTop: "0.75rem",
    paddingTop: "0.625rem",
    borderTop: "1px solid var(--border)",
  },
  ".cm-doc-code-inline": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    overflowWrap: "anywhere",
  },
  ".cm-doc-strong": { fontWeight: "600" },
  ".cm-doc-code": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    lineHeight: "1.1875rem",
    whiteSpace: "pre",
    overflowX: "auto",
    padding: "0.4375rem 0.625rem",
    borderRadius: "var(--radius-md)",
    backgroundColor: "var(--code)",
  },
  ".cm-doc-math": {
    overflowX: "auto",
    overflowY: "hidden",
    padding: "0.125rem 0",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
    whiteSpace: "pre",
  },
  ".cm-doc-math[data-typeset]": { fontFamily: "inherit", fontSize: "inherit", whiteSpace: "normal" },
  ".cm-doc-math .katex-display": { margin: "0", textAlign: "left" },
  ".cm-doc-math .katex-display > .katex": { textAlign: "left" },
  ".cm-doc-math-inline:not([data-typeset])": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-xs)",
  },
  ".cm-doc-card .katex": { fontSize: "1.04em" },
  ".cm-doc-list": { paddingLeft: "1rem", listStyle: "none" },
  ".cm-doc-list-item": { position: "relative" },
  ".cm-doc-list-item + .cm-doc-list-item": { marginTop: "0.25rem" },
  ".cm-doc-list-item::before": {
    content: '"\\2013"',
    position: "absolute",
    left: "-1rem",
    color: "var(--faint)",
  },
  ".cm-doc-fields": { display: "flex", flexDirection: "column" },
  ".cm-doc-field-name": { display: "flex", alignItems: "baseline", flexWrap: "wrap", columnGap: "0.5rem" },
  ".cm-doc-field-name > .cm-doc-code-inline": { fontWeight: "500" },
  ".cm-doc-field-type": {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--text-2xs)",
    color: "var(--muted-foreground)",
  },
  ".cm-doc-field-text": { margin: "0 0 0 0.875rem", color: "var(--muted-foreground)" },
  ".cm-doc-field-text + .cm-doc-field-name": { marginTop: "0.375rem" },
  ".cm-doc-field-text:empty": { display: "none" },
  ".cm-doc-note": {
    paddingLeft: "0.625rem",
    borderLeft: "2px solid var(--border)",
    color: "var(--muted-foreground)",
  },
  ".cm-doc-note-label": { fontWeight: "600", color: "var(--foreground)", marginRight: "0.375rem" },
  ".cm-doc-truncated": { fontSize: "var(--text-xs)", color: "var(--muted-foreground)" },

  // Diagnostics: a thin wavy underline, the message on hover, nothing in the gutter.
  ".cm-lintRange": { backgroundImage: "none", paddingBottom: "0" },
  ".cm-lintRange-error, .cm-lintRange-warning, .cm-lintRange-info": {
    backgroundImage: "none",
    textDecorationLine: "underline",
    textDecorationStyle: "wavy",
    textDecorationThickness: "1px",
    textUnderlineOffset: "3px",
    textDecorationSkipInk: "none",
  },
  ".cm-lintRange-error": { textDecorationColor: "color-mix(in oklch, var(--fail) 75%, transparent)" },
  ".cm-lintRange-warning": { textDecorationColor: "color-mix(in oklch, var(--warn) 75%, transparent)" },
  ".cm-lintRange-info": { textDecorationColor: "var(--faint)" },
  ".cm-lintRange-active": { backgroundColor: "transparent" },
  // A mistake with no width (something is missing here): a small mark under the gap.
  ".cm-lintPoint:after": { borderBottomColor: "var(--fail)", left: "-3px", borderWidth: "0 3px 3px" },
  ".cm-lintPoint-warning:after": { borderBottomColor: "var(--warn)" },
  ".cm-lintPoint-info:after": { borderBottomColor: "var(--faint)" },
  ".cm-tooltip.cm-tooltip-lint": { padding: "0" },
  ".cm-diagnosticText": { display: "flex", alignItems: "baseline", gap: "0.5rem" },
  ".cm-diagnostic": {
    padding: "0.375rem 0.75rem",
    margin: "0",
    border: "none",
    fontFamily: "var(--font-sans)",
    fontSize: "var(--text-xs)",
    lineHeight: "1.125rem",
    whiteSpace: "normal",
    maxWidth: "30rem",
  },
  ".cm-diagnostic + .cm-diagnostic": { borderTop: "1px solid var(--border)" },
  ".cm-diagnostic-label": { flex: "none", fontWeight: "600" },
  ".cm-diagnostic-error .cm-diagnostic-label": { color: "var(--fail)" },
  ".cm-diagnostic-warning .cm-diagnostic-label": { color: "var(--warn)" },
  ".cm-diagnostic-info .cm-diagnostic-label": { color: "var(--muted-foreground)" },
  ".cm-diagnostic-text": { minWidth: "0", overflowWrap: "anywhere" },
});
