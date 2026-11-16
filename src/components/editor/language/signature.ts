import { completionStatus } from "@codemirror/autocomplete";
import { LSPPlugin } from "@codemirror/lsp-client";
import { type Extension, Prec, StateEffect, StateField } from "@codemirror/state";
import {
  type Command,
  type EditorView,
  keymap,
  showTooltip,
  type Tooltip,
  ViewPlugin,
  type ViewUpdate,
} from "@codemirror/view";
import { markupText } from "@/lib/editor/completion";
import { parseInline } from "@/lib/editor/doc-format";
import { docCard } from "./doc-view";

interface ServerParameter {
  label: string | [number, number];
  documentation?: unknown;
}

interface ServerSignature {
  label: string;
  parameters?: ServerParameter[];
  activeParameter?: number | null;
}

interface ServerSignatureHelp {
  signatures: ServerSignature[];
  activeSignature?: number | null;
  activeParameter?: number | null;
}

interface SignatureContext {
  /** 1 invoked, 2 trigger character, 3 content change. */
  triggerKind: 1 | 2 | 3;
  triggerCharacter?: string;
  isRetrigger: boolean;
  activeSignatureHelp?: ServerSignatureHelp;
}

interface Shown {
  help: ServerSignatureHelp;
  active: number;
  pos: number;
}

/** Following the cursor inside a call: how long after a move the server is asked again. */
const FOLLOW_DELAY_MS = 200;

const setSignature = StateEffect.define<Shown | null>();

/** The active parameter of a signature as a range of its label, or null. */
export function activeParameterRange(
  signature: ServerSignature,
  fallback: number | null | undefined,
): [number, number] | null {
  const index = signature.activeParameter ?? fallback;
  const parameter = index != null ? signature.parameters?.[index] : undefined;
  if (!parameter) return null;
  if (Array.isArray(parameter.label)) return parameter.label;
  const start = signature.label.indexOf(parameter.label);
  return start < 0 ? null : [start, start + parameter.label.length];
}

function draw(view: EditorView, { help, active }: Shown): HTMLElement {
  const signature = help.signatures[active] ?? help.signatures[0];
  const card = docCard(view, {}, "cm-doc-card-signature");
  if (!signature) return card;
  const label = document.createElement("div");
  label.className = "cm-doc-signature";
  const range = activeParameterRange(signature, help.activeParameter);
  if (range) {
    const current = document.createElement("span");
    current.className = "cm-doc-active-parameter";
    current.textContent = signature.label.slice(range[0], range[1]);
    label.append(signature.label.slice(0, range[0]), current, signature.label.slice(range[1]));
  } else {
    label.textContent = signature.label;
  }
  card.append(label);

  // Under the signature: what the argument being typed is, not the whole docstring (hover has that).
  const index = signature.activeParameter ?? help.activeParameter;
  const parameter = index != null ? signature.parameters?.[index] : undefined;
  const about = parameter ? markupText(parameter.documentation) : null;
  if (about?.trim()) {
    const note = document.createElement("div");
    note.className = "cm-doc-parameter";
    for (const part of parseInline(about)) {
      if (part.kind === "code") {
        const code = document.createElement("code");
        code.className = "cm-doc-code-inline";
        code.textContent = part.text;
        note.append(code);
      } else note.append(part.text);
    }
    card.append(note);
  }
  if (help.signatures.length > 1) {
    const count = document.createElement("div");
    count.className = "cm-doc-caption";
    count.textContent = `${active + 1} of ${help.signatures.length}`;
    card.append(count);
  }
  return card;
}

const signatureField = StateField.define<{ shown: Shown; tooltip: Tooltip } | null>({
  create: () => null,
  update(value, transaction) {
    for (const effect of transaction.effects) {
      if (!effect.is(setSignature)) continue;
      const shown = effect.value;
      return shown
        ? { shown, tooltip: { pos: shown.pos, above: true, create: (view) => ({ dom: draw(view, shown) }) } }
        : null;
    }
    if (value && transaction.docChanged) {
      const pos = transaction.changes.mapPos(value.shown.pos);
      return { shown: { ...value.shown, pos }, tooltip: { ...value.tooltip, pos } };
    }
    return value;
  },
  provide: (field) => showTooltip.from(field, (value) => value?.tooltip ?? null),
});

function sameHelp(a: ServerSignatureHelp, b: ServerSignatureHelp, active: number): boolean {
  if (a.signatures.length !== b.signatures.length) return false;
  if (!a.signatures.every((signature, index) => signature.label === b.signatures[index]?.label)) return false;
  return (
    (a.signatures[active]?.activeParameter ?? a.activeParameter) ===
    (b.signatures[active]?.activeParameter ?? b.activeParameter)
  );
}

const signaturePlugin = ViewPlugin.fromClass(
  class {
    private request: { dropped: boolean } | null = null;
    private follow: ReturnType<typeof setTimeout> | null = null;

    update(update: ViewUpdate) {
      const plugin = LSPPlugin.get(update.view);
      if (!plugin) return;
      const current = update.state.field(signatureField);
      const provider = plugin.client.serverCapabilities?.signatureHelpProvider;
      let trigger = "";
      if (update.docChanged && update.transactions.some((tr) => tr.isUserEvent("input.type"))) {
        const triggers = [
          ...(provider?.triggerCharacters ?? ["(", ","]),
          ...(current ? (provider?.retriggerCharacters ?? []) : []),
        ];
        update.changes.iterChanges((_fromA, _toA, _fromB, _toB, inserted) => {
          const text = inserted.toString();
          for (const character of triggers) if (text.includes(character)) trigger = character;
        });
      }
      if (trigger) {
        this.ask(plugin, {
          triggerKind: 2,
          triggerCharacter: trigger,
          isRetrigger: current !== null,
          activeSignatureHelp: current?.shown.help,
        });
      } else if (current && (update.selectionSet || update.docChanged)) {
        // The cursor moved inside (or out of) the call: let the server say which argument it is on now.
        if (this.follow) clearTimeout(this.follow);
        this.follow = setTimeout(() => {
          this.follow = null;
          const shown = update.view.state.field(signatureField, false)?.shown;
          if (shown) {
            this.ask(plugin, { triggerKind: 3, isRetrigger: true, activeSignatureHelp: shown.help });
          }
        }, FOLLOW_DELAY_MS);
      }
    }

    ask(plugin: LSPPlugin, context: SignatureContext) {
      if (this.follow) clearTimeout(this.follow);
      this.follow = null;
      if (this.request) this.request.dropped = true;
      const request = { dropped: false };
      this.request = request;
      const { view } = plugin;
      const pos = view.state.selection.main.head;
      plugin.client.sync();
      plugin.client
        .request<unknown, ServerSignatureHelp | null>("textDocument/signatureHelp", {
          textDocument: { uri: plugin.uri },
          position: plugin.toPosition(pos),
          context,
        })
        .then(
          (help) => {
            if (request.dropped) return;
            const current = view.state.field(signatureField, false);
            if (!help?.signatures.length) {
              if (current) view.dispatch({ effects: setSignature.of(null) });
              return;
            }
            const active = Math.min(help.activeSignature ?? 0, help.signatures.length - 1);
            if (current && current.shown.active === active && sameHelp(current.shown.help, help, active))
              return;
            // Keep the card where the call started while moving between its arguments.
            const anchored = current && current.shown.help.signatures[0]?.label === help.signatures[0]?.label;
            view.dispatch({
              effects: setSignature.of({ help, active, pos: anchored ? current.shown.pos : pos }),
            });
          },
          () => {
            // No answer: leave what is shown; the next keystroke asks again.
          },
        );
    }

    destroy() {
      if (this.follow) clearTimeout(this.follow);
      if (this.request) this.request.dropped = true;
    }
  },
);

/** Asks for signature help at the cursor (Mod-Shift-Space). */
const showSignature: Command = (view) => {
  const plugin = LSPPlugin.get(view);
  const signatures = view.plugin(signaturePlugin);
  if (!plugin || !signatures) return false;
  const current = view.state.field(signatureField, false);
  signatures.ask(plugin, {
    triggerKind: 1,
    isRetrigger: Boolean(current),
    activeSignatureHelp: current?.shown.help,
  });
  return true;
};

/** Escape closes the signature card, once the completion list (which Escape closes first) is gone. */
const closeSignature: Command = (view) => {
  if (!view.state.field(signatureField, false) || completionStatus(view.state) !== null) return false;
  view.dispatch({ effects: setSignature.of(null) });
  return true;
};

/**
 * Signature help: typing `(` or `,` in a call shows the signature above the
 * cursor with the argument being typed set apart, and what that argument is
 * when the server knows. Own implementation on the client's request API, so
 * the card is drawn like the others and no documentation is rendered as HTML.
 */
export function signatureHelp(): Extension {
  return [
    signatureField,
    signaturePlugin,
    Prec.high(
      keymap.of([
        { key: "Mod-Shift-Space", run: showSignature },
        { key: "Escape", run: closeSignature },
      ]),
    ),
  ];
}
