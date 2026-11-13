import { createCssVariablesTheme, createHighlighterCoreSync, type HighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import json from "shiki/langs/json.mjs";
import python from "shiki/langs/python.mjs";
import shell from "shiki/langs/shellscript.mjs";

/**
 * Static code highlighting. The theme emits CSS variables (see styles/editor.css)
 * instead of colors, so one render is correct in light and dark and matches the editor.
 */
const theme = createCssVariablesTheme({
  name: "whetstone",
  variablePrefix: "--shiki-",
  fontStyle: true,
});

const ALIASES: Record<string, string> = {
  py: "python",
  python: "python",
  json: "json",
  jsonc: "json",
  sh: "shellscript",
  bash: "shellscript",
  shell: "shellscript",
  shellscript: "shellscript",
};

let highlighter: HighlighterCore | null = null;

function getHighlighter(): HighlighterCore {
  highlighter ??= createHighlighterCoreSync({
    themes: [theme],
    langs: [python, json, shell],
    engine: createJavaScriptRegexEngine(),
  });
  return highlighter;
}

export interface HighlightToken {
  content: string;
  color?: string;
  italic: boolean;
  bold: boolean;
}

/** Tokenizes code into lines of styled spans. Unknown languages come back as plain text. */
export function highlight(code: string, language: string | undefined): HighlightToken[][] {
  const lang = language ? ALIASES[language.toLowerCase()] : undefined;
  if (!lang) {
    return code.split("\n").map((line) => [{ content: line, italic: false, bold: false }]);
  }
  const { tokens } = getHighlighter().codeToTokens(code, { lang, theme: "whetstone" });
  return tokens.map((line) =>
    line.map((token) => ({
      content: token.content,
      color: token.color,
      italic: ((token.fontStyle ?? 0) & 1) !== 0,
      bold: ((token.fontStyle ?? 0) & 2) !== 0,
    })),
  );
}
