import { Link, type LinkProps } from "@tanstack/react-router";
import { cn } from "cn";
import type { ReactNode } from "react";
import type { RouteTarget } from "@/lib/routes";
import { InlineButton } from "./inline-button";

/**
 * In-app navigation. Composes the Button primitive with the router's Link so
 * focus, press, and disabled behavior match every other control.
 *
 * `tone="text"` is an inline text link; `tone="quiet"` reads as plain text until hovered;
 * `stretched` makes the link cover its nearest positioned ancestor (clickable rows).
 */
export function AppLink({
  tone = "text",
  stretched = false,
  truncate = false,
  className,
  children,
  to,
}: {
  /** Build targets with the helpers in lib/routes. */
  to: RouteTarget;
  tone?: "text" | "quiet";
  stretched?: boolean;
  /**
   * Keep the link on one line with an ellipsis; the parent must constrain its width.
   * A text label is repeated as the native tooltip so nothing is hidden for good.
   */
  truncate?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <InlineButton
      nativeButton={false}
      title={truncate && typeof children === "string" ? children : undefined}
      className={cn(
        "justify-start",
        truncate && "block truncate text-left whitespace-nowrap",
        tone === "quiet" && "text-foreground no-underline hover:text-primary hover:no-underline",
        stretched && "static after:absolute after:inset-0",
        className,
      )}
      render={<Link {...(to as unknown as LinkProps)} />}
    >
      {children}
    </InlineButton>
  );
}
