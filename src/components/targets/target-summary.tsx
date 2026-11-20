import { MetaItem, MetaList } from "@/components/app/meta-list";
import type { RemoteDoctor } from "@/lib/api";
import { describeGpu, describeMachine, installedFrameworks } from "@/lib/targets";

/**
 * What a machine is, from its last check: the platform, the GPU and its
 * memory, the Python version, the frameworks installed there, and, for a
 * remote machine, the folder Learn AI keeps its files in. Renders nothing
 * when none of it is known.
 */
export function TargetSummary({
  doctor,
  remoteDir,
  className,
}: {
  /** Null when no check has got as far as reading the machine. */
  doctor: RemoteDoctor | null;
  /** The folder on a remote machine; absent for this computer. */
  remoteDir?: string | null;
  className?: string;
}) {
  if (!doctor && !remoteDir) return null;
  const frameworks = doctor ? installedFrameworks(doctor) : [];
  return (
    <MetaList layout="rows" className={className}>
      {doctor ? (
        <>
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
        </>
      ) : null}
      {remoteDir ? (
        <MetaItem label="Folder there">
          <code className="selectable font-mono">{remoteDir}</code>
        </MetaItem>
      ) : null}
    </MetaList>
  );
}
