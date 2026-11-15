import { MetaItem, MetaList } from "@/components/app/meta-list";
import type { RemoteDoctor } from "@/lib/api";
import { describeGpu, describeMachine, installedFrameworks } from "@/lib/targets";

/**
 * What a machine is, from its last check: the platform, the GPU and its
 * memory, the Python version, and the frameworks installed there.
 */
export function TargetSummary({ doctor, className }: { doctor: RemoteDoctor; className?: string }) {
  const frameworks = installedFrameworks(doctor);
  return (
    <MetaList className={className}>
      <MetaItem label="Machine">
        <span className="selectable">{describeMachine(doctor)}</span>
      </MetaItem>
      <MetaItem label="GPU">
        <span className="selectable">{describeGpu(doctor) ?? "None"}</span>
      </MetaItem>
      <MetaItem label="Python">
        <span className="selectable tabular">{doctor.python}</span>
      </MetaItem>
      {frameworks.length > 0 ? (
        <MetaItem label="Frameworks">
          <span className="selectable tabular">
            {frameworks.map(({ label, version }) => `${label} ${version}`).join(", ")}
          </span>
        </MetaItem>
      ) : null}
    </MetaList>
  );
}
