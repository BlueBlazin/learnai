import { RuledList } from "@/components/app/ruled-list";
import type { SshHost, TargetInfo } from "@/lib/api";
import type { TargetCheckState } from "@/lib/api/target-check-state";
import { AddTargetForm } from "./add-target-form";
import { TargetRow } from "./target-row";

/**
 * Where the judge can run: this computer first, then the remote machines that
 * were added, each with its last check; and the form that adds another from
 * the SSH configuration.
 */
export function TargetsPanel({
  targets,
  hosts,
  checks,
  adding = false,
  removing,
  onCheck,
  onAdd,
  onRemove,
}: {
  targets: TargetInfo[];
  hosts: SshHost[];
  /** Check state by target name. */
  checks: Record<string, TargetCheckState>;
  adding?: boolean;
  /** The target being removed, if any. */
  removing?: string | null;
  onCheck: (name: string) => void;
  onAdd: (host: string) => void;
  onRemove: (name: string) => void;
}) {
  return (
    <div className="flex flex-col gap-5">
      <RuledList aria-label="Run targets">
        {targets.map((target) => (
          <TargetRow
            key={target.name}
            target={target}
            check={checks[target.name]}
            removing={removing === target.name}
            onCheck={() => onCheck(target.name)}
            onRemove={target.kind === "ssh" ? () => onRemove(target.name) : undefined}
          />
        ))}
      </RuledList>
      <AddTargetForm
        hosts={hosts}
        taken={targets.flatMap((target) => target.host ?? [])}
        adding={adding}
        onAdd={onAdd}
      />
    </div>
  );
}
