import { cn } from "cn";
import type { ReactNode } from "react";

/**
 * Preformatted text from a run: a traceback, an expected-versus-got excerpt,
 * captured prints, an install log. Scrolls in both directions past `maxHeight`.
 */
export function OutputBlock({
  label,
  children,
  tone = "neutral",
  size = "default",
  className,
}: {
  label?: string;
  children: ReactNode;
  tone?: "neutral" | "fail";
  /** `tall` is for logs that are the main content of their panel. */
  size?: "default" | "tall";
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      {label ? <p className="text-xs font-medium text-muted-foreground">{label}</p> : null}
      <pre
        // biome-ignore lint/a11y/noNoninteractiveTabindex: scrollable regions must be keyboard reachable.
        tabIndex={0}
        className={cn(
          "selectable overflow-auto rounded-md bg-code px-3 py-2 font-mono text-xs whitespace-pre",
          size === "tall" ? "max-h-80" : "max-h-56",
          tone === "fail" ? "text-fail" : "text-foreground",
        )}
      >
        {children}
      </pre>
    </div>
  );
}
