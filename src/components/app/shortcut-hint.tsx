import { cn } from "cn";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { ENTER_KEY, MOD_KEY, SHIFT_KEY } from "@/lib/platform";

export type ShortcutKey = "mod" | "shift" | "enter" | (string & {});

const GLYPH: Record<string, string> = { mod: MOD_KEY, shift: SHIFT_KEY, enter: ENTER_KEY };

export const SHORTCUTS = {
  run: ["mod", "enter"],
  submit: ["shift", "mod", "enter"],
  palette: ["mod", "K"],
  toggleStatement: ["mod", "B"],
  toggleResults: ["mod", "J"],
} satisfies Record<string, ShortcutKey[]>;

/** A keyboard shortcut rendered as key caps, optionally with the action it performs. */
export function ShortcutHint({
  keys,
  label,
  className,
}: {
  keys: ShortcutKey[];
  label?: string;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs text-muted-foreground", className)}>
      {label ? <span>{label}</span> : null}
      <KbdGroup>
        {keys.map((key) => (
          <Kbd key={key}>{GLYPH[key] ?? key}</Kbd>
        ))}
      </KbdGroup>
    </span>
  );
}
