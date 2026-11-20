import { SettingSwitch } from "./setting-switch";

/** Turns spaced re-solve on or off. While it is off, the row says what became of the schedules. */
export function ReviewSetting({
  enabled,
  disabled = false,
  onChange,
}: {
  /** Undefined while the stored value is not known yet. */
  enabled: boolean | undefined;
  /** True while the change is being saved. */
  disabled?: boolean;
  onChange: (enabled: boolean) => void;
}) {
  return (
    <SettingSwitch
      label="Reviews"
      help={
        enabled === false
          ? "Hidden. Their schedules are kept for when you turn this back on."
          : "Solved problems come back after 3, 7, 21, 60, and 180 days."
      }
      checked={enabled}
      disabled={disabled}
      onChange={onChange}
    />
  );
}
