import { CircleCheck } from "lucide-react";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import type { EditorStatus } from "@/lib/api";
import type { EditorInstallState } from "@/lib/api/use-editor-install";
import type { EditorSessionView } from "@/lib/editor";
import { suggestionsState, suggestionsVersion } from "@/lib/editor/status";
import { InstallButton } from "./install-actions";
import { InstallLog } from "./install-log";
import { SettingSwitch } from "./setting-switch";

const QUIET_STATUS = {
  installed: "Installed. It starts when you open a problem.",
  starting: "Starting",
  running: "Running",
  restarting: "Restarting",
} as const;

/**
 * Suggestions and documentation in the editor: the switch, whether the
 * language server behind them is installed and running, its version, and the
 * Install action with its log when it is missing.
 */
export function SuggestionsPanel({
  status,
  session,
  saving = false,
  onToggle,
  install,
  onInstall,
  onRetry,
}: {
  status: EditorStatus;
  session: EditorSessionView;
  /** True while the preference is being saved. */
  saving?: boolean;
  onToggle: (enabled: boolean) => void;
  install: EditorInstallState;
  onInstall: () => void;
  /** Starts the server again after it failed. */
  onRetry: () => void;
}) {
  const state = suggestionsState(status, session);
  const version = suggestionsVersion(status, session);
  const installing = install.status === "running" || status.installing;
  return (
    <div className="flex flex-col gap-6">
      <SettingSwitch
        label="Suggest and document as I type"
        checked={status.enabled}
        disabled={saving}
        onChange={onToggle}
      >
        While you write a solution, the editor completes names, shows a function's signature and documentation
        when you point at it or call it, and underlines mistakes such as a misspelled name. It all comes from
        ty, a Python language server that runs on this computer.
      </SettingSwitch>

      {state.kind === "off" ? null : (
        <MetaList layout="rows" className="text-sm">
          <MetaItem label="Status">
            {state.kind === "notInstalled" ? (
              <Notice
                action={
                  <InstallButton installing={installing} disabled={installing} onClick={onInstall}>
                    Install
                  </InstallButton>
                }
              >
                Not installed. The download is about 14 MB.
              </Notice>
            ) : state.kind === "failed" ? (
              <Notice
                tone="fail"
                action={
                  <Button variant="outline" size="xs" onClick={onRetry}>
                    Try again
                  </Button>
                }
              >
                {state.message}
              </Notice>
            ) : (
              <span className="flex items-center gap-2">
                {state.kind === "running" ? <CircleCheck className="size-3.5 text-pass" aria-hidden /> : null}
                <span className="selectable">{QUIET_STATUS[state.kind]}</span>
              </span>
            )}
          </MetaItem>
          {version ? (
            <MetaItem label="ty">
              <span className="selectable tabular">{version}</span>
            </MetaItem>
          ) : null}
        </MetaList>
      )}

      <InstallLog
        label="Install log, ty"
        status={install.status}
        log={install.log}
        message={install.message}
        succeeded="Installed. Suggestions start the next time you open a problem."
      />
    </div>
  );
}
