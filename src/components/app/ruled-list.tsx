import { cn } from "cn";
import type { ReactNode } from "react";

/**
 * A list whose rows are separated by hairlines: submissions, cases, tracks,
 * hints. `framed` also rules it off above and below. Rows are plain `<li>`s.
 */
export function RuledList({
  ordered = false,
  framed = true,
  className,
  children,
  "aria-label": label,
}: {
  /** Render an `<ol>`: the order of the rows carries meaning. */
  ordered?: boolean;
  framed?: boolean;
  className?: string;
  children: ReactNode;
  "aria-label"?: string;
}) {
  const List = ordered ? "ol" : "ul";
  return (
    <List
      aria-label={label}
      className={cn(
        "flex flex-col divide-y divide-border/70",
        framed && "border-y border-border/70",
        className,
      )}
    >
      {children}
    </List>
  );
}
