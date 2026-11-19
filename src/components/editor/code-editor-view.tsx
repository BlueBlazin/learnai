import {
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
} from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { bracketMatching, indentOnInput, indentUnit } from "@codemirror/language";
import { diagnosticCount, setDiagnostics } from "@codemirror/lint";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
import { Annotation, Compartment, EditorState } from "@codemirror/state";
import {
  drawSelection,
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
  type Rect,
  tooltips,
} from "@codemirror/view";
import { cn } from "cn";
import { useEffect, useRef, useState } from "react";
import { type EditorDocument, useEditorSession, useEditorSessionUser } from "@/lib/editor";
import { placeInfo } from "@/lib/editor/info-position";
import { documentKey, documentUri } from "@/lib/editor/uri";
import { languageSupport } from "./language";
import { learnaiEditorTheme } from "./theme";

/** Where floating layers may go: the window, less a small margin so none touches its edge. */
const WINDOW_MARGIN = 8;
function windowSpace() {
  return {
    top: WINDOW_MARGIN,
    left: WINDOW_MARGIN,
    bottom: window.innerHeight - WINDOW_MARGIN,
    right: window.innerWidth - WINDOW_MARGIN,
  };
}

/**
 * Places the documentation panel beside the suggestion list (see `placeInfo`). The
 * list sits above the cursor when the window has no room for it below.
 */
function positionInfo(
  view: EditorView,
  list: Rect,
  option: Rect,
  info: Rect,
  space: Rect,
): { style: string; class: string } {
  const cursor = view.coordsAtPos(view.state.selection.main.head);
  const above = cursor !== null && list.bottom <= (cursor.top + cursor.bottom) / 2;
  const placed = placeInfo({ list, option, info, space, above });
  return {
    style: `${placed.side}: ${placed.offset}px; max-width: ${placed.maxWidth}px; max-height: ${placed.maxHeight}px`,
    class: `cm-completionInfo-${placed.where}`,
  };
}

/** Marks transactions that come from the `value` prop so they are not echoed to `onChange`. */
const External = Annotation.define<boolean>();

export interface CodeEditorProps {
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
  /** Accessible name, for example "Solution, PyTorch". */
  label: string;
  /**
   * The solution file this editor holds. With it (and unless `readOnly`) the
   * editor gets suggestions, hover documentation, signature help, and
   * diagnostics from the language server. Without it the editor is plain.
   */
  document?: EditorDocument;
  className?: string;
}

/** The CodeMirror view behind `CodeEditor`. Import `CodeEditor` instead; this module is heavy. */
export function CodeEditorView({
  value,
  onChange,
  readOnly = false,
  label,
  document: file,
  className,
}: CodeEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const latestValue = useRef(value);
  latestValue.current = value;
  /** Holds what the language server adds; empty while there is no connection. */
  const [language] = useState(() => new Compartment());

  // One language server for the whole app: this editor only says that it wants it.
  const assisted = !readOnly && file !== undefined;
  useEditorSessionUser(assisted);
  const session = useEditorSession();
  const connection = assisted && session.status === "ready" ? session.connection : null;
  const fileKey = documentKey(file);
  const uri = connection && file ? documentUri(connection.rootUri, file) : null;

  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: latestValue.current,
        extensions: [
          lineNumbers(),
          highlightActiveLine(),
          highlightActiveLineGutter(),
          history(),
          drawSelection(),
          indentOnInput(),
          bracketMatching(),
          closeBrackets(),
          autocompletion({ icons: false, positionInfo }),
          // The editor sits in a clipped, resizable pane. Mounting the floating layers on
          // the body lets a list or card opened near the pane's edge extend past it, and
          // CodeMirror then flips and shifts them against the window instead.
          tooltips({ parent: document.body, tooltipSpace: windowSpace }),
          language.of([]),
          highlightSelectionMatches(),
          python(),
          indentUnit.of("    "),
          EditorState.tabSize.of(4),
          EditorState.readOnly.of(readOnly),
          EditorView.editable.of(!readOnly),
          EditorView.contentAttributes.of({ "aria-label": label, spellcheck: "false" }),
          keymap.of([
            ...closeBracketsKeymap,
            ...defaultKeymap.filter((binding) => binding.key !== "Mod-Enter"),
            ...historyKeymap,
            ...searchKeymap,
            ...completionKeymap,
            indentWithTab,
          ]),
          learnaiEditorTheme,
          EditorView.updateListener.of((update) => {
            if (!update.docChanged) return;
            if (update.transactions.some((tr) => tr.annotation(External))) return;
            onChangeRef.current?.(update.state.doc.toString());
          }),
        ],
      }),
    });
    setView(editor);
    // The mono face is bundled but loads on first use. CodeMirror caches character
    // and line metrics, so measure again once the real font is in.
    let live = true;
    void document.fonts?.ready.then(() => live && editor.requestMeasure());
    return () => {
      live = false;
      editor.destroy();
      setView((current) => (current === editor ? null : current));
    };
  }, [readOnly, label, language]);

  // Adopt an external value (file changed on disk, reset, framework switch). The
  // change is an ordinary transaction, so the language server hears about it too.
  useEffect(() => {
    if (!view) return;
    const current = view.state.doc.toString();
    if (current === value) return;
    view.dispatch({
      changes: { from: 0, to: current.length, insert: value },
      annotations: External.of(true),
    });
  }, [view, value]);

  // Connect the document to the language server. Runs after the value has been
  // adopted, so a new document opens with its own text. A new server process
  // (`connection`) or another problem or framework (`uri`) replaces the
  // extension: the old document is closed on its server, the new one opened.
  // biome-ignore lint/correctness/useExhaustiveDependencies: fileKey stands for the document.
  useEffect(() => {
    if (!view) return;
    // Underlines belong to the document and the server that produced them.
    if (diagnosticCount(view.state) > 0) view.dispatch(setDiagnostics(view.state, []));
    view.dispatch({
      effects: language.reconfigure(connection && uri ? languageSupport(connection, uri) : []),
    });
  }, [view, language, connection, uri, fileKey]);

  return <div ref={host} className={cn("code-editor selectable min-h-0 bg-paper", className)} />;
}
