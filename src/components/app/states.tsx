import { cn } from "cn";
import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

/** Nothing to show yet. Says what will appear here and, when possible, how to make it appear. */
export function EmptyState({
  title,
  children,
  action,
  className,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-start gap-1.5 py-6", className)}>
      <p className="font-serif text-md text-foreground">{title}</p>
      {children ? <div className="max-w-measure text-sm text-muted-foreground">{children}</div> : null}
      {action ? <div className="pt-2">{action}</div> : null}
    </div>
  );
}

/** Something failed. Shows the message from the core verbatim and offers a retry when one exists. */
export function ErrorState({
  title = "Something went wrong",
  error,
  onRetry,
  className,
}: {
  title?: string;
  error: unknown;
  onRetry?: () => void;
  className?: string;
}) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className={cn("flex flex-col items-start gap-1.5 py-6", className)}>
      <p className="flex items-center gap-2 font-serif text-md text-foreground">
        <TriangleAlert className="size-4 text-warn" aria-hidden />
        {title}
      </p>
      <p className="selectable max-w-measure font-mono text-xs text-muted-foreground">{message}</p>
      {onRetry ? (
        <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  );
}

const WIDTHS = ["w-11/12", "w-full", "w-4/5", "w-full", "w-2/3", "w-5/6"];

/** Placeholder lines while text loads. */
export function LoadingLines({ lines = 4, className }: { lines?: number; className?: string }) {
  return (
    <div
      className={cn("flex flex-col gap-3 py-2", className)}
      role="status"
      aria-busy="true"
      aria-label="Loading"
    >
      {Array.from({ length: lines }, (_, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: static placeholder list.
        <Skeleton key={i} className={cn("h-3.5", WIDTHS[i % WIDTHS.length])} />
      ))}
    </div>
  );
}
