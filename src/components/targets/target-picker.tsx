import { Server } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { TargetInfo } from "@/lib/api";

/**
 * Chooses where Run and Submit execute: this computer or a remote machine.
 * Shown only when there is a choice to make.
 */
export function TargetPicker({
  targets,
  value,
  disabled = false,
  onChange,
}: {
  targets: TargetInfo[];
  /** The chosen target's name. */
  value: string;
  disabled?: boolean;
  onChange: (name: string) => void;
}) {
  if (targets.length < 2) return null;
  const items = targets.map((target) => ({ value: target.name, label: target.label }));
  return (
    <Select items={items} value={value} onValueChange={(next) => next && onChange(next as string)}>
      <SelectTrigger size="sm" aria-label="Run on" disabled={disabled} className="max-w-44 gap-1.5">
        <Server className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end" alignItemWithTrigger={false} className="min-w-44">
        <SelectGroup>
          {items.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
