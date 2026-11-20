import type { ReactNode } from "react";
import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { EditorStatus, EnvStatus, Framework } from "@/lib/api";
import type { EditorInstallState } from "@/lib/api/use-editor-install";
import type { InstallState } from "@/lib/api/use-env-install";
import type { EditorSessionView } from "@/lib/editor";
import { suggestionsState, suggestionsVersion } from "@/lib/editor/status";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { InstallButton, UvGuideButton } from "./install-actions";
import { InstallLog } from "./install-log";
import { RowStatus } from "./row-status";
import { SettingList, SettingRow } from "./setting-row";

const SESSION_WORD = { starting: "Starting", restarting: "Restarting" } as const;

/** The language server behind suggestions: what the app knows about it, and how to install or restart it. */
export interface LanguageServer {
  /** Undefined while the stored status is not known yet. */
  status: EditorStatus | undefined;
  session: EditorSessionView;
  install: EditorInstallState;
  onInstall: () => void;
  /** Starts the server again after it failed. */
  onRetry: () => void;
}

/**
 * The tools the app runs on, a row each: uv, Python, the frameworks, and ty
 * (the language server). Each has its version or what is missing, and the
 * action that fixes it. An install streams its log under its own row while it
 * runs and after it failed.
 */
export function EnvironmentPanel({
  env,
  install,
  onInstall,
  server,
}: {
  env: EnvStatus;
  install: InstallState;
  /** Installs a framework; `torch` also sets up the base environment. */
  onInstall: (framework: Framework) => void;
  server: LanguageServer;
}) {
  const busy = install.status === "running";
  // Without uv nothing can be installed.
  const blocked = env.uv === null;
  // Setting up Python is the PyTorch install: its log belongs to whichever row is still missing.
  const logOwner = install.framework === "torch" && env.python === null ? "python" : install.framework;
  const logFor = (row: Framework) =>
    logOwner === row && (install.status === "running" || install.status === "failed") ? (
      <InstallLog status={install.status} log={install.log} message={install.message} />
    ) : null;
  const installAction = (row: Framework, target: Framework, label: string) => (
    <InstallButton
      installing={busy && logOwner === row}
      disabled={busy || blocked}
      onClick={() => onInstall(target)}
    >
      {install.status === "failed" && logOwner === row ? "Try again" : label}
    </InstallButton>
  );

  return (
    <SettingList aria-label="Environment">
      <SettingRow
        label="uv"
        help={env.uv ? undefined : "Install uv, then reopen Learn AI."}
        status={
          env.uv ? <RowStatus tone="pass">{env.uv}</RowStatus> : <RowStatus tone="warn">Not found</RowStatus>
        }
      >
        {env.uv ? null : <UvGuideButton />}
      </SettingRow>
      <SettingRow
        label="Python"
        help={env.python || blocked ? undefined : "The first run sets it up, or do it now."}
        status={
          env.python ? (
            <RowStatus tone="pass">{env.python}</RowStatus>
          ) : (
            <RowStatus tone={blocked ? "idle" : "warn"}>{blocked ? "Needs uv" : "Not set up"}</RowStatus>
          )
        }
        body={logFor("python")}
      >
        {env.python || blocked ? null : installAction("python", "torch", "Set up")}
      </SettingRow>
      {env.frameworks.map((entry) =>
        // Python has its own row above; it comes with the environment, not as an install.
        entry.framework === "python" ? null : (
          <SettingRow
            key={entry.framework}
            label={FRAMEWORK_LABEL[entry.framework]}
            status={
              entry.installed ? (
                <RowStatus tone="pass">{entry.version ?? "Installed"}</RowStatus>
              ) : (
                <RowStatus tone="idle">Not installed</RowStatus>
              )
            }
            body={logFor(entry.framework)}
          >
            {entry.installed ? null : installAction(entry.framework, entry.framework, "Install")}
          </SettingRow>
        ),
      )}
      <LanguageServerRow server={server} />
    </SettingList>
  );
}

function LanguageServerRow({
  server: { status, session, install, onInstall, onRetry },
}: {
  server: LanguageServer;
}) {
  const help = "The language server behind suggestions and docs.";
  if (!status) {
    return <SettingRow label="ty" help={help} status={<Skeleton className="h-3.5 w-16" />} />;
  }
  const installing = install.status === "running" || status.installing;
  const version = suggestionsVersion(status, session);
  // While suggestions are off the session has nothing to say; the row still says whether ty is there.
  const state = status.enabled ? suggestionsState(status, session) : null;
  let mark: ReactNode;
  let action: ReactNode = null;
  let failure: ReactNode = null;
  if (!status.installed) {
    mark = <RowStatus tone="idle">Not installed</RowStatus>;
    action = (
      <InstallButton installing={installing} disabled={installing} onClick={onInstall}>
        {install.status === "failed" ? "Try again" : "Install"}
      </InstallButton>
    );
  } else if (state?.kind === "failed") {
    mark = <RowStatus tone="fail">Stopped</RowStatus>;
    action = (
      <Button variant="outline" size="xs" onClick={onRetry}>
        Try again
      </Button>
    );
    failure = <Notice tone="fail">{state.message}</Notice>;
  } else if (state?.kind === "starting" || state?.kind === "restarting") {
    mark = <RowStatus tone="busy">{SESSION_WORD[state.kind]}</RowStatus>;
  } else {
    mark = <RowStatus tone="pass">{version ?? "Installed"}</RowStatus>;
  }
  const log =
    install.status === "running" || install.status === "failed" ? (
      <InstallLog status={install.status} log={install.log} message={install.message} />
    ) : null;
  return (
    <SettingRow
      label="ty"
      help={status.installed ? help : `${help} About 14 MB.`}
      status={mark}
      body={
        log || failure ? (
          <>
            {log}
            {failure}
          </>
        ) : null
      }
    >
      {action}
    </SettingRow>
  );
}
