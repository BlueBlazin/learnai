import {
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
} from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { bracketMatching, indentOnInput, indentUnit } from "@codemirror/language";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
import { Annotation, EditorState } from "@codemirror/state";
import {
  drawSelection,
  EditorView,
  highlightActiveLine,
  highlightActiveLineGutter,
  keymap,
  lineNumbers,
} from "@codemirror/view";
import { cn } from "cn";
import { useEffect, useRef } from "react";
import { whetstoneEditorTheme } from "./theme";

/** Marks transactions that come from the `value` prop so they are not echoed to `onChange`. */
const External = Annotation.define<boolean>();

export interface CodeEditorProps {
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
  /** Accessible name, for example "Solution, PyTorch". */
  label: string;
  className?: string;
}

/** The CodeMirror view behind `CodeEditor`. Import `CodeEditor` instead; this module is heavy. */
export function CodeEditorView({ value, onChange, readOnly = false, label, className }: CodeEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const initialValue = useRef(value);
  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: initialValue.current,
        extensions: [
          lineNumbers(),
          highlightActiveLine(),
          highlightActiveLineGutter(),
          history(),
          drawSelection(),
          indentOnInput(),
          bracketMatching(),
          closeBrackets(),
          autocompletion({ icons: false }),
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
          whetstoneEditorTheme,
          EditorView.updateListener.of((update) => {
            if (!update.docChanged) return;
            if (update.transactions.some((tr) => tr.annotation(External))) return;
            onChangeRef.current?.(update.state.doc.toString());
          }),
        ],
      }),
    });
    view.current = editor;
    // The mono face is bundled but loads on first use. CodeMirror caches character
    // and line metrics, so measure again once the real font is in.
    let live = true;
    void document.fonts?.ready.then(() => live && editor.requestMeasure());
    return () => {
      live = false;
      editor.destroy();
      view.current = null;
    };
  }, [readOnly, label]);

  // Adopt an external value (file changed on disk, reset, framework switch).
  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    if (current === value) return;
    editor.dispatch({
      changes: { from: 0, to: current.length, insert: value },
      annotations: External.of(true),
    });
  }, [value]);

  return <div ref={host} className={cn("code-editor selectable min-h-0 bg-paper", className)} />;
}
