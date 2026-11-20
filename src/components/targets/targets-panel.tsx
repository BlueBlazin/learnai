import { Plus } from "lucide-react";
import { useState } from "react";
import { SettingList } from "@/components/settings/setting-row";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { SshHost, TargetInfo } from "@/lib/api";
import type { TargetCheckState } from "@/lib/api/target-check-state";
import { AddTargetForm } from "./add-target-form";
import { TargetRow } from "./target-row";

/**
 * Where the judge can run: this computer first, then the remote machines that
 * were added, a row each. "Add a machine" opens the form that adds another
 * from the SSH configuration, and closes again once one has been added.
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
  const [addOpen, setAddOpen] = useState(false);
  // A machine that was just added closes the form: its row is the confirmation.
  const [count, setCount] = useState(targets.length);
  if (count !== targets.length) {
    setCount(targets.length);
    if (targets.length > count) setAddOpen(false);
  }
  return (
    <div className="flex flex-col gap-4">
      <SettingList aria-label="Run targets">
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
      </SettingList>
      <Collapsible open={addOpen} onOpenChange={setAddOpen} className="flex flex-col items-start gap-3">
        <CollapsibleTrigger render={<Button variant="outline" size="sm" />}>
          <Plus data-icon="inline-start" />
          Add a machine
        </CollapsibleTrigger>
        <CollapsibleContent className="w-full">
          <AddTargetForm
            hosts={hosts}
            taken={targets.flatMap((target) => target.host ?? [])}
            adding={adding}
            onAdd={onAdd}
          />
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
