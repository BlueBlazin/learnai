import { CircleCheck, LoaderCircle, TriangleAlert } from "lucide-react";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { OutputBlock } from "@/components/results/output-block";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { EnvStatus, Framework } from "@/lib/api";
import type { InstallState } from "@/lib/api/use-env-install";
import { FRAMEWORK_LABEL } from "@/lib/labels";

function Version({ value, missing }: { value: string | null; missing: string }) {
  return value ? (
    <span className="selectable tabular">{value}</span>
  ) : (
    <span className="flex items-center gap-1.5 text-warn">
      <TriangleAlert className="size-3.5" aria-hidden />
      {missing}
    </span>
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
  return (
    <div className="flex flex-col gap-6">
      <MetaList layout="rows" className="text-sm">
        <MetaItem label="uv">
          <Version value={env.uv} missing="Not found. Install uv, then reopen Whetstone." />
        </MetaItem>
        <MetaItem label="Python">
          <Version value={env.python} missing="Not found. uv installs it on the first run." />
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
                  {entry.installed ? null : (
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busy}
                      onClick={() => onInstall(entry.framework)}
                    >
                      {installing ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : null}
                      {installing ? "Installing" : `Install ${FRAMEWORK_LABEL[entry.framework]}`}
                    </Button>
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
            <p className="flex items-center gap-2 text-sm text-pass">
              <CircleCheck className="size-3.5" aria-hidden />
              {FRAMEWORK_LABEL[install.framework]} installed.
            </p>
          ) : null}
          {install.status === "failed" ? (
            <p className="flex items-center gap-2 text-sm text-fail">
              <TriangleAlert className="size-3.5" aria-hidden />
              {install.message ?? "The install failed. The log above has the details."}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
