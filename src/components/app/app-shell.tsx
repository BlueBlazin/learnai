import { Link, type LinkProps } from "@tanstack/react-router";
import { cn } from "cn";
import { Search } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { TITLEBAR_OVERLAY } from "@/lib/platform";
import { type RouteTarget, routes } from "@/lib/routes";
import { useCommandPalette } from "@/stores/workspace-actions";
import { SHORTCUTS, ShortcutHint } from "./shortcut-hint";

function NavLink({ to, exact = false, children }: { to: RouteTarget; exact?: boolean; children: ReactNode }) {
  return (
    <Button
      nativeButton={false}
      variant="ghost"
      size="sm"
      className="no-drag relative h-titlebar rounded-none px-1 font-medium text-muted-foreground hover:bg-transparent data-[status=active]:text-foreground data-[status=active]:after:absolute data-[status=active]:after:inset-x-1 data-[status=active]:after:bottom-0 data-[status=active]:after:h-0.5 data-[status=active]:after:bg-foreground"
      render={<Link {...(to as unknown as LinkProps)} activeOptions={{ exact, includeSearch: false }} />}
    >
      {children}
    </Button>
  );
}

/**
 * The window frame: one quiet bar with the wordmark, the three destinations,
 * and the command palette. The bar is the window's drag region when the
 * title bar is overlaid.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const openPalette = useCommandPalette((state) => state.setOpen);
  return (
    <div className="flex h-full flex-col">
      <header
        data-tauri-drag-region
        className={cn(
          "drag-region flex h-titlebar shrink-0 items-center gap-6 border-b bg-background pr-3",
          TITLEBAR_OVERLAY ? "pl-24" : "pl-4",
        )}
      >
        <Link
          {...(routes.home() as unknown as LinkProps)}
          className="no-drag rounded-sm font-serif text-md font-medium tracking-tight italic"
          aria-label="Whetstone home"
        >
          Whetstone
        </Link>
        <nav aria-label="Main" className="flex items-center gap-4">
          <NavLink to={routes.home()} exact>
            Home
          </NavLink>
          <NavLink to={routes.library()}>Library</NavLink>
          <NavLink to={routes.settings()}>Settings</NavLink>
          {import.meta.env.DEV ? <NavLink to={routes.devComponents()}>Components</NavLink> : null}
        </nav>
        <Button
          variant="outline"
          size="sm"
          className="no-drag ml-auto w-56 justify-start gap-2 bg-paper font-normal text-muted-foreground"
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
