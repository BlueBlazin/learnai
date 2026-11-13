import { Play, RotateCcw } from "lucide-react";
import { ActionButton } from "@/components/app/action-button";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { IconButton } from "@/components/app/icon-button";
import { SHORTCUTS } from "@/components/app/shortcut-hint";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { CodeEditor } from "@/components/editor/code-editor";
import { FrameworkSwitcher } from "@/components/problem/framework-switcher";
import type { Framework, FrameworkEnv } from "@/lib/api";
import type { SolutionEditor } from "@/lib/api/use-solution-editor";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { SaveIndicator } from "./save-indicator";

/**
 * The right-hand top panel: a toolbar (framework, save state, reset, Run,
 * Submit) over the code editor.
 */
export function EditorPane({
  frameworks,
  framework,
  solvedFrameworks,
  env,
  editor,
  busy,
  onFrameworkChange,
  onRun,
  onSubmit,
}: {
  frameworks: Framework[];
  framework: Framework;
  solvedFrameworks: Framework[];
  env?: FrameworkEnv[];
  editor: SolutionEditor;
  /** True while a run is in flight: Run and Submit are unavailable. */
  busy: boolean;
  onFrameworkChange: (framework: Framework) => void;
  onRun: () => void;
  onSubmit: () => void;
}) {
  const ready = editor.code !== undefined;
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-toolbar shrink-0 items-center gap-2 border-b bg-background px-3">
        <FrameworkSwitcher
          frameworks={frameworks}
          value={framework}
          onChange={onFrameworkChange}
          solved={solvedFrameworks}
          env={env}
          disabled={busy}
        />
        <SaveIndicator state={editor.saveState} className="pl-1" />
        <div className="ml-auto flex items-center gap-1.5">
          <ConfirmDialog
            trigger={
              <IconButton label="Reset to starter code" disabled={!ready || editor.isStarter || busy}>
                <RotateCcw />
              </IconButton>
            }
            title="Reset to the starter code?"
            description={`Your ${FRAMEWORK_LABEL[framework]} solution for this problem will be replaced with the starter. Submitted code stays in your submission history.`}
            confirmLabel="Reset code"
            destructive
            onConfirm={() => void editor.reset()}
          />
          <ActionButton
            variant="outline"
            size="sm"
            hint="Run the public cases"
            shortcut={SHORTCUTS.run}
            disabled={!ready || busy}
            onClick={onRun}
          >
            <Play data-icon="inline-start" />
            Run
          </ActionButton>
          <ActionButton
            size="sm"
            hint="Run every case and record the attempt"
            shortcut={SHORTCUTS.submit}
            disabled={!ready || busy}
            onClick={onSubmit}
          >
            Submit
          </ActionButton>
        </div>
      </div>
      <div className="min-h-0 flex-1 bg-paper">
        {editor.error ? (
          <ErrorState title="The solution file could not be read" error={editor.error} className="px-4" />
        ) : editor.code === undefined ? (
          <LoadingLines lines={6} className="p-4" />
        ) : (
          <CodeEditor
            value={editor.code}
            onChange={editor.setCode}
            label={`Solution, ${FRAMEWORK_LABEL[framework]}`}
          />
        )}
      </div>
    </div>
  );
}
