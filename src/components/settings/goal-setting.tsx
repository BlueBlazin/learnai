import { useId } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { pluralize } from "@/lib/format";
import { SettingRow } from "./setting-row";

/** The goals on offer. A goal saved from elsewhere that is not here still shows as the current value. */
export const GOAL_CHOICES = [3, 5, 7, 10, 14];

/** Chooses how many problems to aim to solve each week. */
export function GoalSetting({ value, onChange }: { value: number; onChange: (goal: number) => void }) {
  const id = useId();
  const choices = GOAL_CHOICES.includes(value)
    ? GOAL_CHOICES
    : [...GOAL_CHOICES, value].sort((a, b) => a - b);
  const items = choices.map((choice) => ({ value: String(choice), label: pluralize(choice, "problem") }));
  return (
    <SettingRow label="Weekly goal" labelFor={id} help="Problems to solve each week.">
      <Select items={items} value={String(value)} onValueChange={(next) => next && onChange(Number(next))}>
        <SelectTrigger id={id} size="sm" className="min-w-32 gap-1.5">
          <SelectValue />
        </SelectTrigger>
        <SelectContent align="end" alignItemWithTrigger={false} className="min-w-32">
          <SelectGroup>
            {items.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </SettingRow>
  );
}
