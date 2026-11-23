import {
  Award,
  BookOpen,
  House,
  ListOrdered,
  Monitor,
  Moon,
  PanelBottom,
  PanelLeft,
  Play,
  RotateCw,
  Send,
  Settings,
  Sun,
} from "lucide-react";
import { useEffect, useMemo } from "react";
import { StatusMark } from "@/components/problem/status-mark";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { useGo } from "@/hooks/use-go";
import { useProblems, useReviews, useTracks } from "@/lib/api/queries";
import { isModKey } from "@/lib/platform";
import { nextDueReview } from "@/lib/reviews";
import { type RouteTarget, routes } from "@/lib/routes";
import { useCommandPalette } from "@/stores/command-palette";
import { type ThemePreference, useThemeStore } from "@/stores/theme";
import { useWorkspaceActions } from "@/stores/workspace-actions";
import { SHORTCUTS, ShortcutHint } from "./shortcut-hint";

/** Every whitespace-separated term must appear in the item; stricter and more predictable than fuzzy matching. */
function matchAllTerms(value: string, search: string): number {
  const haystack = value.toLowerCase();
  return search
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => haystack.includes(term))
    ? 1
    : 0;
}

/**
 * The command palette (Mod-K): jump to any problem, move between screens,
 * switch theme, run or submit and show or hide the statement and the results
 * when a problem is open, and start the next review when one is due.
 */
export function CommandPalette() {
  const { open, setOpen } = useCommandPalette();
  const go = useGo();
  const problems = useProblems();
  const tracks = useTracks();
  const reviews = useReviews();
  const nextReview = nextDueReview(reviews.data ?? []);
  const nextReviewTitle = nextReview
    ? problems.data?.find((problem) => problem.id === nextReview.problemId)?.title
    : undefined;
  const workspace = useWorkspaceActions((state) => state.actions);
  const setTheme = useThemeStore((state) => state.setPreference);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isModKey(event) && !event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(!useCommandPalette.getState().open);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setOpen]);

  const trackTitle = useMemo(
    () => new Map((tracks.data ?? []).map((track) => [track.id, track.title])),
    [tracks.data],
  );

  const close = () => setOpen(false);
  const navigate = (target: RouteTarget) => {
    close();
    go(target);
  };
  const act = (action: () => void) => {
    close();
    action();
  };
  const theme = (preference: ThemePreference) => act(() => setTheme(preference));

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      description="Jump to a problem or run a command."
      className="sm:max-w-xl"
    >
      <Command filter={matchAllTerms}>
        <CommandInput placeholder="Jump to a problem or run a command" />
        <CommandList className="max-h-96">
          <CommandEmpty>Nothing matches. Try a title, a tag, or a track.</CommandEmpty>
          {workspace ? (
            <CommandGroup heading={workspace.problemTitle}>
              <CommandItem disabled={workspace.busy} onSelect={() => act(workspace.run)}>
                <Play />
                Run public cases
                <CommandShortcut>
                  <ShortcutHint keys={SHORTCUTS.run} />
                </CommandShortcut>
              </CommandItem>
              <CommandItem disabled={workspace.busy} onSelect={() => act(workspace.submit)}>
                <Send />
                Submit against all cases
                <CommandShortcut>
                  <ShortcutHint keys={SHORTCUTS.submit} />
                </CommandShortcut>
              </CommandItem>
              <CommandItem onSelect={() => act(workspace.toggleStatement)}>
                <PanelLeft />
                {workspace.statementCollapsed ? "Show statement" : "Hide statement"}
                <CommandShortcut>
                  <ShortcutHint keys={SHORTCUTS.toggleStatement} />
                </CommandShortcut>
              </CommandItem>
              <CommandItem onSelect={() => act(workspace.toggleResults)}>
                <PanelBottom />
                {workspace.resultsCollapsed ? "Show results" : "Hide results"}
                <CommandShortcut>
                  <ShortcutHint keys={SHORTCUTS.toggleResults} />
                </CommandShortcut>
              </CommandItem>
            </CommandGroup>
          ) : null}
          {nextReview && nextReviewTitle ? (
            <CommandGroup heading="Review">
              <CommandItem
                value={`Start next review ${nextReviewTitle}`}
                onSelect={() => navigate(routes.problem(nextReview.problemId, { review: true }))}
              >
                <RotateCw />
                Start next review
                <span className="ml-auto shrink-0 truncate pl-4 text-xs text-muted-foreground">
                  {nextReviewTitle}
                </span>
              </CommandItem>
            </CommandGroup>
          ) : null}
          <CommandGroup heading="Go to">
            <CommandItem onSelect={() => navigate(routes.home())}>
              <House />
              Home
            </CommandItem>
            <CommandItem onSelect={() => navigate(routes.curriculum())}>
              <ListOrdered />
              Curriculum
            </CommandItem>
            <CommandItem onSelect={() => navigate(routes.accomplishments())}>
              <Award />
              Accomplishments
            </CommandItem>
            <CommandItem onSelect={() => navigate(routes.library())}>
              <BookOpen />
              Library
            </CommandItem>
            <CommandItem onSelect={() => navigate(routes.settings())}>
              <Settings />
              Settings
            </CommandItem>
          </CommandGroup>
          <CommandGroup heading="Theme">
            <CommandItem onSelect={() => theme("light")}>
              <Sun />
              Use light theme
            </CommandItem>
            <CommandItem onSelect={() => theme("dark")}>
              <Moon />
              Use dark theme
            </CommandItem>
            <CommandItem onSelect={() => theme("system")}>
              <Monitor />
              Match the system theme
            </CommandItem>
          </CommandGroup>
          <CommandGroup heading="Problems">
            {(problems.data ?? []).map((problem) => (
              <CommandItem
                key={problem.id}
                value={`${problem.title} ${problem.tags.join(" ")} ${trackTitle.get(problem.track) ?? ""} ${problem.id}`}
                onSelect={() => navigate(routes.problem(problem.id))}
              >
                <StatusMark status={problem.status} />
                <span className="truncate">{problem.title}</span>
                <span className="ml-auto shrink-0 pl-4 text-xs text-muted-foreground">
                  {trackTitle.get(problem.track)}
                </span>
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  );
}
