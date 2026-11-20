import { Link, type LinkProps } from "@tanstack/react-router";
import { cn } from "cn";
import { TriangleAlert } from "lucide-react";
import { SETTINGS_SECTION_LABEL } from "@/lib/labels";
import { routes, SETTINGS_SECTIONS, type SettingsSection } from "@/lib/routes";

/**
 * The categories of Settings, as links: the current one is marked the way the
 * app marks a selected row, and a category that needs the user carries a
 * small warning mark after its name.
 */
export function SettingsNav({
  current,
  attention,
  className,
}: {
  current: SettingsSection;
  /** The categories that need the user (`settingsAttention`). */
  attention: ReadonlySet<SettingsSection>;
  className?: string;
}) {
  return (
    <nav aria-label="Settings" className={cn("flex flex-col gap-0.5", className)}>
      {SETTINGS_SECTIONS.map((section) => {
        const active = section === current;
        return (
          <Link
            key={section}
            {...(routes.settings(section) as unknown as LinkProps)}
            activeOptions={{ exact: true }}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex h-7 items-center justify-between gap-2 rounded-md px-2.5 text-sm transition-colors hover:bg-accent",
              active ? "bg-accent font-semibold" : "text-foreground",
            )}
          >
            {SETTINGS_SECTION_LABEL[section]}
            {attention.has(section) ? (
              <span className="inline-flex">
                <TriangleAlert className="size-3.5 text-warn" aria-hidden />
                <span className="sr-only">, needs attention</span>
              </span>
            ) : null}
          </Link>
        );
      })}
    </nav>
  );
}
