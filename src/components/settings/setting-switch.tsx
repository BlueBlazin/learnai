import { type ReactNode, useId } from "react";
import { Switch } from "@/components/ui/switch";

/** A preference that is on or off: the switch, what it controls, and the sentence that explains it. */
export function SettingSwitch({
  label,
  children,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  /** The explanation under the label. */
  children: ReactNode;
  checked: boolean;
  /** True while the change is being saved. */
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="flex max-w-measure items-start gap-3">
      <Switch
        id={id}
        className="mt-0.5"
        checked={checked}
        disabled={disabled}
        onCheckedChange={(next) => onChange(next)}
      />
      <div className="flex min-w-0 flex-col gap-1">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        <p className="selectable text-xs text-pretty text-muted-foreground">{children}</p>
      </div>
    </div>
  );
}
