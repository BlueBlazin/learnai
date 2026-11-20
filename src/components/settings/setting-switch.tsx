import { type ReactNode, useId } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { SettingRow } from "./setting-row";

/** A preference that is on or off, as a row: its name, one line of help, and the switch. */
export function SettingSwitch({
  label,
  help,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  help?: ReactNode;
  /** Undefined while the stored value is not known yet. */
  checked: boolean | undefined;
  /** True while the change is being saved. */
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <SettingRow label={label} labelFor={id} help={help}>
      {checked === undefined ? (
        <Skeleton className="h-4.5 w-8" />
      ) : (
        <Switch id={id} checked={checked} disabled={disabled} onCheckedChange={(next) => onChange(next)} />
      )}
    </SettingRow>
  );
}
