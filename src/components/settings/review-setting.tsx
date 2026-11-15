import { useId } from "react";
import { Switch } from "@/components/ui/switch";

/** Turns spaced re-solve on or off, with the one sentence that says what it does. */
export function ReviewSetting({
  enabled,
  disabled = false,
  onChange,
}: {
  enabled: boolean;
  /** True while the change is being saved. */
  disabled?: boolean;
  onChange: (enabled: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="flex max-w-measure items-start gap-3">
      <Switch
        id={id}
        className="mt-0.5"
        checked={enabled}
        disabled={disabled}
        onCheckedChange={(checked) => onChange(checked)}
      />
      <div className="flex min-w-0 flex-col gap-1">
        <label htmlFor={id} className="text-sm font-medium">
          Ask me to solve problems again
        </label>
        <p className="selectable text-xs text-pretty text-muted-foreground">
          A solved problem comes back for review after 3, 7, 21, 60, and 180 days, each time from the blank
          starter. Turning this off hides reviews and keeps their schedules.
        </p>
      </div>
    </div>
  );
}
