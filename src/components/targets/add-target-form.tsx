import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { SshHost } from "@/lib/api";
import { FixText } from "./fix-text";

const EXAMPLE_HOST = `Host gpu-box
  HostName 192.168.1.20
  User your-name`;

function describe(host: SshHost): string {
  const where = [host.user ? `${host.user}@` : "", host.hostName ?? ""].join("");
  return where ? `${host.alias} (${where})` : host.alias;
}

/**
 * Adds a remote machine as a run target: a choice among the hosts in
 * `~/.ssh/config` that are not targets yet, and what adding one does. With no
 * host to offer, says what to add to the configuration instead. Shown when
 * "Add a machine" is opened, so it carries no heading of its own.
 */
export function AddTargetForm({
  hosts,
  taken,
  adding = false,
  onAdd,
}: {
  /** Every host alias in the SSH configuration. */
  hosts: SshHost[];
  /** Aliases that are targets already. */
  taken: string[];
  adding?: boolean;
  onAdd: (host: string) => void;
}) {
  const available = hosts.filter((host) => !taken.includes(host.alias));
  const [chosen, setChosen] = useState<string | null>(null);
  const value = available.some((host) => host.alias === chosen) ? chosen : (available[0]?.alias ?? null);
  const note = (
    <p className="selectable text-xs text-pretty text-muted-foreground">
      The machine needs uv. The first check copies the judge to ~/.learnai there and builds its Python
      environment. Keys and passwords stay with SSH and your agent; Learn AI never reads or stores them.
    </p>
  );

  if (available.length === 0) {
    return (
      <div className="flex max-w-measure flex-col gap-2">
        <p className="selectable text-sm text-pretty text-muted-foreground">
          {hosts.length === 0 ? (
            <>
              There is no host in ~/.ssh/config yet. Add an entry like this one, confirm in a terminal that{" "}
              <code className="font-mono text-xs">ssh gpu-box</code> signs in without a password prompt, and
              it appears here.
            </>
          ) : (
            "Every host in ~/.ssh/config is already a target. Add another entry like this one and it appears here."
          )}
        </p>
        <FixText>{EXAMPLE_HOST}</FixText>
        {note}
      </div>
    );
  }

  const items = available.map((host) => ({ value: host.alias, label: describe(host) }));
  return (
    <div className="flex max-w-measure flex-col gap-2">
      <div className="flex items-center gap-2">
        <Select items={items} value={value} onValueChange={(next) => setChosen(next as string | null)}>
          <SelectTrigger size="sm" aria-label="Host from ~/.ssh/config" className="min-w-56">
            <SelectValue />
          </SelectTrigger>
          <SelectContent align="start" alignItemWithTrigger={false} className="min-w-56">
            <SelectGroup>
              {items.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  {item.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
        <Button
          variant="outline"
          size="sm"
          disabled={value === null || adding}
          onClick={() => value && onAdd(value)}
        >
          {adding ? "Adding" : "Add"}
        </Button>
      </div>
      {note}
    </div>
  );
}
