import { Outlet } from "@tanstack/react-router";
import { AppShell } from "@/components/app/app-shell";
import { CommandPalette } from "@/components/app/command-palette";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

export function RootLayout() {
  return (
    <TooltipProvider>
      <AppShell>
        <Outlet />
      </AppShell>
      <CommandPalette />
      <Toaster position="bottom-right" />
    </TooltipProvider>
  );
}
