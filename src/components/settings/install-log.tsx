import { Notice } from "@/components/app/notice";
import { OutputBlock } from "@/components/results/output-block";

/** The streamed log of an install, with how it ended: shown from the moment the install starts. */
export function InstallLog({
  label,
  status,
  log,
  message,
  succeeded,
}: {
  /** Names the log, for example "Install log, MLX". */
  label: string;
  status: "idle" | "running" | "succeeded" | "failed";
  log: string[];
  /** Why it failed, when the installer said. */
  message: string | null;
  /** The sentence shown once it has succeeded. */
  succeeded: string;
}) {
  if (status === "idle") return null;
  return (
    <div className="flex flex-col gap-2" aria-live="polite">
      <OutputBlock label={label} size="tall">
        {log.length > 0 ? log.join("\n") : "Starting"}
      </OutputBlock>
      {status === "succeeded" ? <Notice tone="pass">{succeeded}</Notice> : null}
      {status === "failed" ? (
        <Notice tone="fail">{message ?? "The install failed. The log above has the details."}</Notice>
      ) : null}
    </div>
  );
}
