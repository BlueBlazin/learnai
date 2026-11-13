import { CircleCheck } from "lucide-react";
import type { ReactNode } from "react";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Notice } from "@/components/app/notice";
import { OutputBlock } from "@/components/results/output-block";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { EnvStatus, Framework } from "@/lib/api";
import type { InstallState } from "@/lib/api/use-env-install";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { InstallButton, UvGuideButton } from "./install-actions";

/** A version string, or the reason there is none with the action that fixes it. */
function Version({ value, missing, action }: { value: string | null; missing: string; action?: ReactNode }) {
  return value ? (
    <span className="selectable tabular">{value}</span>
  ) : (
    <Notice action={action}>{missing}</Notice>
  );
}

/**
 * The Python environment: uv, the interpreter, and each framework with its
 * version or an Install action. An install streams its log underneath.
 */
export function EnvironmentPanel({
  env,
  install,
  onInstall,
}: {
  env: EnvStatus;
  install: InstallState;
  onInstall: (framework: Framework) => void;
}) {
  const busy = install.status === "running";
  // Without uv nothing can be installed; say so once, at the top, instead of failing per framework.
  const blocked = env.uv === null;
  return (
    <div className="flex flex-col gap-6">
      <MetaList layout="rows" className="text-sm">
        <MetaItem label="uv">
          <Version
            value={env.uv}
            missing="Not found. Install uv, then reopen Whetstone."
            action={<UvGuideButton />}
          />
        </MetaItem>
        <MetaItem label="Python">
          <Version
            value={env.python}
            missing={
              blocked
                ? "Unavailable until uv is installed."
                : "Not set up yet. The first run sets it up, or do it now."
            }
            action={
              blocked ? null : (
                <InstallButton
                  installing={busy && install.framework === "torch"}
                  disabled={busy}
                  onClick={() => onInstall("torch")}
                >
                  Set up now
                </InstallButton>
              )
            }
          />
        </MetaItem>
      </MetaList>

      <Table>
        <TableHeader>
          <TableRow className="border-rule hover:bg-transparent">
            <TableHead className="pl-0">Framework</TableHead>
            <TableHead>Version</TableHead>
            <TableHead className="pr-0 text-right">
              <span className="sr-only">Action</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {env.frameworks.map((entry) => {
            const installing = busy && install.framework === entry.framework;
            return (
              <TableRow key={entry.framework} className="hover:bg-transparent">
                <TableCell className="w-48 pl-0 font-medium">{FRAMEWORK_LABEL[entry.framework]}</TableCell>
                <TableCell>
                  {entry.installed ? (
                    <span className="flex items-center gap-2">
                      <CircleCheck className="size-3.5 text-pass" aria-hidden />
                      <span className="selectable tabular">{entry.version ?? "Installed"}</span>
                    </span>
                  ) : (
                    <span className="text-muted-foreground">Not installed</span>
                  )}
                </TableCell>
                <TableCell className="pr-0 text-right">
                  {/* Python comes with the environment itself; there is nothing separate to install. */}
                  {entry.installed || entry.framework === "python" ? null : (
                    <InstallButton
                      size="sm"
                      installing={installing}
                      disabled={busy || blocked}
                      onClick={() => onInstall(entry.framework)}
                    >
                      Install {FRAMEWORK_LABEL[entry.framework]}
                    </InstallButton>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      {install.status !== "idle" && install.framework ? (
        <div className="flex flex-col gap-2" aria-live="polite">
          <OutputBlock label={`Install log, ${FRAMEWORK_LABEL[install.framework]}`} size="tall">
            {install.log.length > 0 ? install.log.join("\n") : "Starting"}
          </OutputBlock>
          {install.status === "succeeded" ? (
            <Notice tone="pass">{FRAMEWORK_LABEL[install.framework]} installed.</Notice>
          ) : null}
          {install.status === "failed" ? (
            <Notice tone="fail">
              {install.message ?? "The install failed. The log above has the details."}
            </Notice>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
