import { SettingSwitch } from "./setting-switch";

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
  return (
    <SettingSwitch
      label="Ask me to solve problems again"
      checked={enabled}
      disabled={disabled}
      onChange={onChange}
    >
      A solved problem comes back for review after 3, 7, 21, 60, and 180 days, each time from the blank
      starter. Turning this off hides reviews and keeps their schedules.
    </SettingSwitch>
  );
}
