import { cn } from "cn";

/** A stage's color as a small square beside its name, so the name itself stays in ink. */
export function StageSwatch({ color, className }: { color: string; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block size-2 shrink-0", className)}
      style={{ backgroundColor: color }}
    />
  );
}
