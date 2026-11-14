import { cn } from "cn";
import { CircleCheck, Info, type LucideIcon, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

type NoticeTone = "warn" | "info" | "pass" | "fail";

const ICON: Record<NoticeTone, LucideIcon> = {
  warn: TriangleAlert,
  info: Info,
  pass: CircleCheck,
  fail: TriangleAlert,
};

const ICON_TONE: Record<NoticeTone, string> = {
  warn: "text-warn",
  info: "text-muted-foreground",
  pass: "text-pass",
  fail: "text-fail",
};

/**
 * One line that says something needs attention (or went well, or failed) and,
 * when possible, offers the action that fixes it. A strip under a toolbar
 * (`strip`) or a line in a page (`inline`); never a box. `footer` sits under the
 * line, inside the same strip: the progress of the action once it is under way.
 */
export function Notice({
  tone = "warn",
  layout = "inline",
  children,
  action,
  footer,
  className,
}: {
  tone?: NoticeTone;
  layout?: "inline" | "strip";
  children: ReactNode;
  action?: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  const Icon = ICON[tone];
  return (
    <div
      role="status"
      className={cn(
        "flex flex-col gap-1.5 text-sm",
        layout === "strip" && "shrink-0 border-b bg-background px-3 py-1.5",
        className,
      )}
    >
      <div className="flex items-center gap-2">
        <Icon className={cn("size-3.5 shrink-0", ICON_TONE[tone])} aria-hidden />
        <p className={cn("selectable min-w-0 flex-1", tone === "fail" ? "text-fail" : "text-foreground")}>
          {children}
        </p>
        {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
      </div>
      {footer}
    </div>
  );
}
