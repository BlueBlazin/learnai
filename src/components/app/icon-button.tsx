import type { ComponentProps, ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ShortcutHint, type ShortcutKey } from "./shortcut-hint";

type ButtonProps = ComponentProps<typeof Button>;

/** An icon-only button. The label is required: it is the accessible name and the tooltip. */
export function IconButton({
  label,
  shortcut,
  children,
  size = "icon-sm",
  variant = "ghost",
  side = "bottom",
  ...props
}: Omit<ButtonProps, "aria-label"> & {
  label: string;
  shortcut?: ShortcutKey[];
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
}) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button aria-label={label} size={size} variant={variant} {...props} />}>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side}>
        {label}
        {shortcut ? <ShortcutHint keys={shortcut} /> : null}
      </TooltipContent>
    </Tooltip>
  );
}
