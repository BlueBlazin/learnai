import { Notice } from "@/components/app/notice";
import { OutputBlock } from "@/components/results/output-block";

/**
 * The streamed log of an install, under the row that started it: shown while
 * the install runs and, with the reason, after it failed. An install that
 * succeeded needs no log; its row says so.
 */
export function InstallLog({
  status,
  log,
  message,
}: {
  status: "idle" | "running" | "succeeded" | "failed";
  log: string[];
  /** Why it failed, when the installer said. */
  message: string | null;
}) {
  if (status !== "running" && status !== "failed") return null;
  return (
    <div className="flex flex-col gap-2" aria-live="polite">
      <OutputBlock>{log.length > 0 ? log.join("\n") : "Starting"}</OutputBlock>
      {status === "failed" ? (
        <Notice tone="fail">{message ?? "The install failed. The log above has the details."}</Notice>
      ) : null}
    </div>
  );
}
