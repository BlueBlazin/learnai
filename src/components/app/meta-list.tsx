import { cn } from "cn";
import type { ReactNode } from "react";

/**
 * Label and value pairs. `inline` sets them in a wrapping row, the way a paper
 * sets its metadata strip (problem headers, submission headers); `rows` stacks
 * them with aligned labels (Settings).
 */
export function MetaList({
  children,
  layout = "inline",
  className,
}: {
  children: ReactNode;
  layout?: "inline" | "rows";
  className?: string;
}) {
  return (
    <dl
      data-layout={layout}
      className={cn(
        "group/meta text-xs",
        layout === "inline" ? "flex flex-wrap gap-x-7 gap-y-1.5" : "flex flex-col gap-2.5",
        className,
      )}
    >
      {children}
    </dl>
  );
}

export function MetaItem({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-2 group-data-[layout=rows]/meta:gap-6">
      <dt className="shrink-0 text-muted-foreground group-data-[layout=rows]/meta:w-28">{label}</dt>
      <dd className="min-w-0 text-foreground">{children}</dd>
    </div>
  );
}
