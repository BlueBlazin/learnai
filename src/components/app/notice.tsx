import { cn } from "cn";
import { Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

/**
 * One line that says something needs attention and, when possible, offers the
 * action that fixes it. A strip under a toolbar (`strip`) or a line in a page
 * (`inline`); never a box.
 */
export function Notice({
  tone = "warn",
  layout = "inline",
  children,
  action,
  className,
}: {
  tone?: "warn" | "info";
  layout?: "inline" | "strip";
  children: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  const Icon = tone === "warn" ? TriangleAlert : Info;
  return (
    <div
      role="status"
      className={cn(
        "flex items-center gap-2 text-sm",
        layout === "strip" && "shrink-0 border-b bg-background px-3 py-1.5",
        className,
      )}
    >
      <Icon
        className={cn("size-3.5 shrink-0", tone === "warn" ? "text-warn" : "text-muted-foreground")}
        aria-hidden
      />
      <p className="selectable min-w-0 flex-1 text-foreground">{children}</p>
      {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
    </div>
  );
}
