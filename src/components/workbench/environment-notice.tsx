import { AppLink } from "@/components/app/app-link";
import { Notice } from "@/components/app/notice";
import { InstallButton, UvGuideButton } from "@/components/settings/install-actions";
import type { Framework } from "@/lib/api";
import type { InstallState } from "@/lib/api/use-env-install";
import type { EnvironmentProblem } from "@/lib/environment";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { routes } from "@/lib/routes";

/**
 * Says, right where Run and Submit are, why a run cannot succeed yet (uv is
 * missing, the Python environment was never set up, or the chosen framework is
 * not installed) and offers the fix.
 */
export function EnvironmentNotice({
  problem,
  install,
  onInstall,
  layout = "strip",
}: {
  problem: EnvironmentProblem;
  install: InstallState;
  /** Installs a framework; `torch` also sets up the base environment. */
  onInstall: (framework: Framework) => void;
  /** `strip` under the editor toolbar; `inline` as a line on a page. */
  layout?: "strip" | "inline";
}) {
  if (problem.kind === "uv-missing") {
    return (
      <Notice layout={layout} action={<UvGuideButton />}>
        uv was not found, so solutions cannot run. Install uv, then reopen Whetstone.
      </Notice>
    );
  }

  const target: Framework = problem.kind === "framework-missing" ? problem.framework : "torch";
  const installing = install.status === "running";
  const failed = install.status === "failed" && install.framework === target;
  const label = problem.kind === "framework-missing" ? `Install ${FRAMEWORK_LABEL[target]}` : "Set up now";
  return (
    <Notice
      layout={layout}
      action={
        <>
          {failed ? <AppLink to={routes.settings()}>See the log in Settings</AppLink> : null}
          <InstallButton installing={installing} disabled={installing} onClick={() => onInstall(target)}>
            {failed ? "Try again" : label}
          </InstallButton>
        </>
      }
    >
      {failed
        ? (install.message ?? "The install failed.")
        : problem.kind === "framework-missing"
          ? `${FRAMEWORK_LABEL[target]} is not installed, so ${FRAMEWORK_LABEL[target]} solutions cannot run yet.`
          : "The Python environment is not set up yet. The first run sets it up, which can take a few minutes."}
    </Notice>
  );
}
