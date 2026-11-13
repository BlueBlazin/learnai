import type { ComponentProps, ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ShortcutHint, type ShortcutKey } from "./shortcut-hint";

/** A labeled button whose tooltip teaches its keyboard shortcut. */
export function ActionButton({
  shortcut,
  hint,
  children,
  ...props
}: ComponentProps<typeof Button> & {
  shortcut: ShortcutKey[];
  /** What the action does, shown before the shortcut. */
  hint: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button {...props} />}>{children}</TooltipTrigger>
      <TooltipContent side="bottom">
        {hint}
        <ShortcutHint keys={shortcut} />
      </TooltipContent>
    </Tooltip>
  );
}
