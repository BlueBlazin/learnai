import { cn } from "cn";
import type { ReactNode } from "react";

/**
 * The 40px bar at the top of a workspace pane: tabs on the statement side, run
 * controls on the editor side. It is a size container (`@container`), so a
 * control can drop its words when the pane is narrow.
 */
export function PaneToolbar({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "@container flex h-toolbar shrink-0 items-center gap-2 border-b bg-background px-3",
        className,
      )}
    >
      {children}
    </div>
  );
}
