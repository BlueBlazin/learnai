import { Link, type LinkProps } from "@tanstack/react-router";
import { cn } from "cn";
import { Search } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { TITLEBAR_OVERLAY, useTrafficLightInset } from "@/lib/platform";
import { type RouteTarget, routes } from "@/lib/routes";
import { useCommandPalette } from "@/stores/command-palette";
import { AppLink } from "./app-link";
import { SHORTCUTS, ShortcutHint } from "./shortcut-hint";

/** A destination in the app bar: full bar height, with a rule under the current one. */
function NavLink({ to, exact = false, children }: { to: RouteTarget; exact?: boolean; children: ReactNode }) {
  return (
    <Button
      nativeButton={false}
      variant="ghost"
      size="sm"
      className={cn(
        "relative h-titlebar rounded-none px-1 font-medium text-muted-foreground hover:bg-transparent",
        "after:absolute after:inset-x-1 after:bottom-0 after:h-0.5",
        "data-[status=active]:text-foreground data-[status=active]:after:bg-foreground",
      )}
      render={<Link {...(to as unknown as LinkProps)} activeOptions={{ exact, includeSearch: false }} />}
    >
      {children}
    </Button>
  );
}

/**
 * The window frame: one quiet bar with the wordmark, the four destinations,
 * and the command palette.
 *
 * With an overlay title bar (macOS) the bar is the window's title bar: it keeps
 * clear of the traffic lights and drags the window. `data-tauri-drag-region="deep"`
 * makes every part of the bar that is not a control a drag handle (double-click
 * zooms), while links and buttons stay clickable.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const openPalette = useCommandPalette((state) => state.setOpen);
  const inset = useTrafficLightInset();
  return (
    <div className="flex h-full flex-col">
      <header
        data-tauri-drag-region={TITLEBAR_OVERLAY ? "deep" : undefined}
        className={cn(
          "flex h-titlebar shrink-0 items-center gap-6 border-b bg-background pr-3",
          inset ? "pl-24" : "pl-4",
        )}
      >
        <span className="font-serif text-md font-medium tracking-tight italic">
          <AppLink to={routes.home()} tone="quiet">
            Whetstone
          </AppLink>
        </span>
        <nav aria-label="Main" className="flex items-center gap-4">
          <NavLink to={routes.home()} exact>
            Home
          </NavLink>
          <NavLink to={routes.paths()}>Path</NavLink>
          <NavLink to={routes.library()}>Library</NavLink>
          <NavLink to={routes.settings()}>Settings</NavLink>
          {import.meta.env.DEV ? <NavLink to={routes.devComponents()}>Components</NavLink> : null}
        </nav>
        <Button
          variant="outline"
          size="sm"
          className="ml-auto w-56 justify-start gap-2 bg-paper font-normal text-muted-foreground"
          onClick={() => openPalette(true)}
        >
          <Search aria-hidden />
          <span className="flex-1 text-left">Jump to a problem</span>
          <ShortcutHint keys={SHORTCUTS.palette} />
        </Button>
      </header>
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}
