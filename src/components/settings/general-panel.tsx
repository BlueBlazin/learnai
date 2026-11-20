import type { ThemePreference } from "@/stores/theme";
import { GoalSetting } from "./goal-setting";
import { ReviewSetting } from "./review-setting";
import { SettingList, SettingRow } from "./setting-row";
import { SettingSwitch } from "./setting-switch";
import { ThemePicker } from "./theme-picker";

/** A stored on/off preference: its value (undefined until read), whether a change is being saved, and the setter. */
export interface StoredSwitch {
  value: boolean | undefined;
  saving?: boolean;
  onChange: (value: boolean) => void;
}

/** Every preference of the app, a row each: theme, weekly goal, reviews, and suggestions in the editor. */
export function GeneralPanel({
  theme,
  onThemeChange,
  weeklyGoal,
  onWeeklyGoalChange,
  reviews,
  suggestions,
}: {
  theme: ThemePreference;
  onThemeChange: (theme: ThemePreference) => void;
  weeklyGoal: number;
  onWeeklyGoalChange: (goal: number) => void;
  reviews: StoredSwitch;
  suggestions: StoredSwitch;
}) {
  return (
    <SettingList aria-label="General">
      <SettingRow label="Theme">
        <ThemePicker value={theme} onChange={onThemeChange} />
      </SettingRow>
      <GoalSetting value={weeklyGoal} onChange={onWeeklyGoalChange} />
      <ReviewSetting enabled={reviews.value} disabled={reviews.saving} onChange={reviews.onChange} />
      <SettingSwitch
        label="Suggestions and docs"
        help="Completions, hover docs, and error underlines while you type."
        checked={suggestions.value}
        disabled={suggestions.saving}
        onChange={suggestions.onChange}
      />
    </SettingList>
  );
}
