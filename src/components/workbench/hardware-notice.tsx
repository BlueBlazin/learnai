import { Notice } from "@/components/app/notice";
import type { Requires } from "@/lib/api";
import { REQUIRES_PHRASE } from "@/lib/labels";

/**
 * Says, where Run and Submit are, that this machine lacks the hardware a
 * problem requires: which hardware, what still runs here, and that a solution
 * cannot be accepted until every case runs.
 */
export function HardwareNotice({
  requires,
  layout = "strip",
}: {
  requires: Exclude<Requires, "cpu">;
  layout?: "strip" | "inline";
}) {
  return (
    <Notice tone="info" layout={layout}>
      This problem needs {REQUIRES_PHRASE[requires]}, which this computer does not have. The other cases still
      run here; a solution is accepted only on a machine that runs every case.
    </Notice>
  );
}
