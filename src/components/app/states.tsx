import { cn } from "cn";
import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

/** The shared shape of a state: a serif line, a quieter explanation, an optional action. */
function StateBlock({
  title,
  icon,
  children,
  action,
  className,
  role,
}: {
  title: string;
  icon?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
  role?: "alert";
}) {
  return (
    <div role={role} className={cn("flex flex-col items-start gap-1.5 py-6", className)}>
      <p className="flex items-center gap-2 font-serif text-md text-foreground">
        {icon}
        {title}
      </p>
      {children}
      {action ? <div className="pt-2">{action}</div> : null}
    </div>
  );
}

/** Nothing to show yet. Says what will appear here and, when possible, how to make it appear. */
export function EmptyState({
  title,
  icon,
  children,
  action,
  className,
}: {
  title: string;
  /** A small mark before the title when the state has a kind, for example a lock. */
  icon?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <StateBlock title={title} icon={icon} action={action} className={className}>
      {children ? <div className="max-w-measure text-sm text-muted-foreground">{children}</div> : null}
    </StateBlock>
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
    <StateBlock
      role="alert"
      title={title}
      icon={<TriangleAlert className="size-4 shrink-0 text-warn" aria-hidden />}
      className={className}
      action={
        onRetry ? (
          <Button variant="outline" size="sm" onClick={onRetry}>
            Try again
          </Button>
        ) : null
      }
    >
      <p className="selectable max-w-measure font-mono text-xs text-muted-foreground">{message}</p>
    </StateBlock>
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
