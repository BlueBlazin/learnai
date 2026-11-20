import { cn } from "cn";
import { swatchColor } from "@/lib/module-colors";

/**
 * A module's color as a small square beside its name, so the name itself stays
 * in ink. `color` is the module's own `#rrggbb` (null falls back to the
 * accent); the square darkens it on the light theme until it holds on paper.
 */
export function ModuleSwatch({ color, className }: { color: string | null; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block size-2 shrink-0", className)}
      style={{ backgroundColor: swatchColor(color) }}
    />
  );
}
