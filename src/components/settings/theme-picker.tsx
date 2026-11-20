import { Monitor, Moon, Sun } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { ThemePreference } from "@/stores/theme";

const OPTIONS = [
  { value: "system", label: "Match system", icon: Monitor },
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
] as const;

/** Chooses the color theme: follow the system, or force light or dark. */
export function ThemePicker({
  value,
  onChange,
}: {
  value: ThemePreference;
  onChange: (value: ThemePreference) => void;
}) {
  return (
    <ToggleGroup
      variant="outline"
      size="sm"
      spacing={0}
      aria-label="Theme"
      value={[value]}
      onValueChange={(next) => {
        const choice = (next as ThemePreference[])[0];
        if (choice) onChange(choice);
      }}
    >
      {OPTIONS.map(({ value: option, label, icon: Icon }) => (
        <ToggleGroupItem
          key={option}
          value={option}
          // The choice in force reads by ink as well as by fill: the others step back.
          className="text-muted-foreground aria-pressed:bg-accent aria-pressed:text-foreground"
        >
          <Icon data-icon="inline-start" aria-hidden />
          {label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
