import { cn } from "cn";
import type { ReactNode } from "react";
import { DisclosureChevron } from "@/components/app/disclosure-chevron";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

/**
 * The list a Settings pane is made of: rows separated by hairlines, ruled off
 * above and below. The rows share its three columns (name, status, control),
 * so statuses and controls line up down the list whatever their widths.
 */
export function SettingList({
  children,
  className,
  "aria-label": label,
}: {
  children: ReactNode;
  className?: string;
  "aria-label"?: string;
}) {
  return (
    <ul
      aria-label={label}
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_auto_auto] gap-x-5 divide-y divide-border/70 border-y border-border/70",
        className,
      )}
    >
      {children}
    </ul>
  );
}

/**
 * One thing in Settings, on one line: its name, at most one short line of help
 * under it, and at the right its status and its control. `body` sits under the
 * line while there is something to follow (a log, progress, a failure).
 * `details` is kept behind the name, which then opens and closes it. A row is
 * never a box: no background, no border of its own. Use inside `SettingList`.
 */
export function SettingRow({
  label,
  labelFor,
  help,
  status,
  children,
  body,
  details,
  open,
  onOpenChange,
}: {
  /** The name, in the interface face. */
  label: ReactNode;
  /** The id of the control the name labels, when it is a form control. */
  labelFor?: string;
  /** One short line under the name, in secondary ink. Only where the name is not enough. */
  help?: ReactNode;
  /** Where the thing stands (a version, "Not installed"), in the column before the control. */
  status?: ReactNode;
  /** The control or action, at the right edge. */
  children?: ReactNode;
  /** Shown under the line, always: what is happening now or what went wrong. */
  body?: ReactNode;
  /** Detail on demand. When given, the name becomes the button that discloses it. */
  details?: ReactNode;
  /** Controls the disclosure; uncontrolled (and closed) when absent. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  const nameClass = "text-sm font-medium text-foreground";
  const line = (
    <div className="col-span-full grid min-h-7 grid-cols-subgrid items-center">
      <div className="flex min-w-0 flex-col items-start gap-0.5">
        {details ? (
          <CollapsibleTrigger
            className={cn(
              nameClass,
              "group/disclosure flex max-w-full items-center gap-1.5 rounded-sm text-left",
            )}
          >
            <span className="min-w-0 truncate">{label}</span>
            <DisclosureChevron className="group-hover/disclosure:text-muted-foreground" />
          </CollapsibleTrigger>
        ) : labelFor ? (
          <label htmlFor={labelFor} className={nameClass}>
            {label}
          </label>
        ) : (
          <p className={cn(nameClass, "selectable max-w-full truncate")}>{label}</p>
        )}
        {help ? (
          <p className="selectable max-w-full text-xs text-pretty text-muted-foreground">{help}</p>
        ) : null}
      </div>
      {status ? <div className="flex items-center text-xs">{status}</div> : null}
      <div className={cn("flex items-center justify-end gap-2", !status && "col-span-2")}>{children}</div>
    </div>
  );
  const className = "col-span-full grid grid-cols-subgrid py-2";
  const under = body ? <div className="col-span-full flex flex-col gap-2 pt-2 pb-1">{body}</div> : null;
  if (!details) {
    return (
      <li className={className}>
        {line}
        {under}
      </li>
    );
  }
  return (
    <Collapsible render={<li />} className={className} open={open} onOpenChange={onOpenChange}>
      {line}
      {under}
      <CollapsibleContent className="col-span-full">
        <div className="flex flex-col gap-3 pt-2.5 pb-2">{details}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}
