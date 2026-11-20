import type { ReactNode } from "react";
import { AppLink } from "@/components/app/app-link";
import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import type { Requires } from "@/lib/api";
import { REQUIRES_PHRASE } from "@/lib/labels";
import { routes } from "@/lib/routes";

/**
 * Says, where Run and Submit are, that the machine a run would use lacks the
 * hardware a problem requires: which hardware, what still runs there, and that
 * a solution cannot be accepted until every case runs. Then it points to the
 * way out: a remote machine known to have the hardware (`remote`, with a
 * button that switches to it), or, when none is known, adding one.
 */
export function HardwareNotice({
  requires,
  machine,
  remote,
  onUseRemote,
  layout = "strip",
}: {
  requires: Exclude<Requires, "cpu">;
  /** The remote machine a run would use, by name; absent for this computer. */
  machine?: string;
  /** A remote machine whose last check found the hardware, by name. */
  remote?: string;
  /** Switches the workspace to `remote`. */
  onUseRemote?: () => void;
  layout?: "strip" | "inline";
}) {
  const where = machine ? (
    <>
      which <span className="font-medium">{machine}</span> does not have
    </>
  ) : (
    "which this computer does not have"
  );
  let way: ReactNode;
  let action: ReactNode = null;
  if (remote) {
    way = (
      <>
        {" "}
        <span className="font-medium">{remote}</span> has it.
      </>
    );
    action = onUseRemote ? (
      <Button variant="outline" size="xs" onClick={onUseRemote}>
        Run on {remote}
      </Button>
    ) : null;
  } else {
    way = " A remote machine that has it can run every case.";
    action = <AppLink to={routes.settings("targets")}>Add a remote machine</AppLink>;
  }
  return (
    <Notice tone="info" layout={layout} action={action}>
      This problem needs {REQUIRES_PHRASE[requires]}, {where}. The other cases still run; a solution is
      accepted only where every case runs.{way}
    </Notice>
  );
}
