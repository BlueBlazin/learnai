import { Play, RotateCcw } from "lucide-react";
import type { ReactNode } from "react";
import { ActionButton } from "@/components/app/action-button";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { IconButton } from "@/components/app/icon-button";
import { SHORTCUTS } from "@/components/app/shortcut-hint";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { CodeEditor } from "@/components/editor/code-editor";
import { FrameworkSwitcher } from "@/components/problem/framework-switcher";
import { Separator } from "@/components/ui/separator";
import type { EnvStatus, Framework } from "@/lib/api";
import type { InstallState } from "@/lib/api/use-env-install";
import type { SolutionEditor } from "@/lib/api/use-solution-editor";
import { environmentProblem } from "@/lib/environment";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { EnvironmentNotice } from "./environment-notice";
import { PaneToolbar } from "./pane-toolbar";
import { SaveIndicator } from "./save-indicator";

/**
 * The right-hand top panel: a toolbar (layout controls, framework, save state,
 * reset, where to run, Run, Submit) over the code editor. This pane is the one
 * that is always showing, so `layoutControls` (the toggles that show and hide
 * the panels around it) sit at the start of its toolbar, apart from the run controls. When the environment cannot run the chosen
 * framework, a notice under the toolbar says why and offers the fix; `notices`
 * adds what else stands between a run and an accepted verdict. In a review
 * the same pane edits and judges the review attempt. The editor gets
 * suggestions and documentation for the file it holds; `suggestions` says so
 * in the toolbar when they are not available.
 */
export function EditorPane({
  problemId,
  frameworks,
  framework,
  solvedFrameworks,
  env,
  install,
  onInstall,
  editor,
  busy,
  onFrameworkChange,
  onRun,
  onSubmit,
  notices,
  suggestions,
  layoutControls,
  targetPicker,
  remote = false,
  review = false,
}: {
  /**
   * The editor holds a review attempt: the strip under the toolbar (passed in
   * `notices`) owns starting over, so the reset control is not shown here.
   */
  review?: boolean;
  problemId: string;
  frameworks: Framework[];
  framework: Framework;
  solvedFrameworks: Framework[];
  /** Environment status, when known. */
  env?: EnvStatus;
  install: InstallState;
  onInstall: (framework: Framework) => void;
  editor: SolutionEditor;
  /** True while a run is in flight: Run and Submit are unavailable. */
  busy: boolean;
  onFrameworkChange: (framework: Framework) => void;
  onRun: () => void;
  onSubmit: () => void;
  /** More strips under the toolbar: a model that is not downloaded, hardware this machine lacks. */
  notices?: ReactNode;
  /** The state of suggestions and documentation, shown only when it needs attention. */
  suggestions?: ReactNode;
  /** The toggles that show and hide the statement and the results. */
  layoutControls?: ReactNode;
  /** The control that chooses where runs execute, when there is a choice. */
  targetPicker?: ReactNode;
  /** True when runs go to a remote machine: the local environment then does not matter. */
  remote?: boolean;
}) {
  const ready = editor.code !== undefined;
  const problem = remote ? null : environmentProblem(env, framework);
  return (
    <div className="flex h-full flex-col">
      <PaneToolbar>
        {layoutControls ? (
          <>
            <div className="-ml-1.5 flex items-center">{layoutControls}</div>
            <Separator orientation="vertical" className="my-2.5" />
          </>
        ) : null}
        <FrameworkSwitcher
          frameworks={frameworks}
          value={framework}
          onChange={onFrameworkChange}
          solved={solvedFrameworks}
          env={env?.frameworks}
          disabled={busy}
        />
        <SaveIndicator state={editor.saveState} className="pl-1" />
        {suggestions}
        <div className="ml-auto flex items-center gap-1.5">
          {review ? null : (
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
          )}
          {targetPicker}
          <ActionButton
            variant="outline"
            size="sm"
            hint={review ? "Run the public cases on your review attempt" : "Run the public cases"}
            shortcut={SHORTCUTS.run}
            disabled={!ready || busy}
            onClick={onRun}
          >
            <Play data-icon="inline-start" />
            Run
          </ActionButton>
          <ActionButton
            size="sm"
            hint={
              review
                ? "Run every case; an accepted review moves the schedule on"
                : "Run every case and record the attempt"
            }
            shortcut={SHORTCUTS.submit}
            disabled={!ready || busy}
            onClick={onSubmit}
          >
            Submit
          </ActionButton>
        </div>
      </PaneToolbar>
      {problem ? <EnvironmentNotice problem={problem} install={install} onInstall={onInstall} /> : null}
      {notices}
      <div className="min-h-0 flex-1 bg-paper">
        {editor.error ? (
          <ErrorState title="The solution file could not be read" error={editor.error} className="px-4" />
        ) : editor.code === undefined ? (
          <LoadingLines lines={6} className="p-4" />
        ) : (
          <CodeEditor
            value={editor.code}
            onChange={editor.setCode}
            label={`${review ? "Review attempt" : "Solution"}, ${FRAMEWORK_LABEL[framework]}`}
            document={{ problemId, framework, review }}
          />
        )}
      </div>
    </div>
  );
}
