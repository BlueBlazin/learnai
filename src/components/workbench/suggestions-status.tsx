import { LoaderCircle, TriangleAlert } from "lucide-react";
import { AppLink } from "@/components/app/app-link";
import { InlineButton } from "@/components/app/inline-button";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { EditorSessionView } from "@/lib/editor";
import { routes } from "@/lib/routes";

/** In a narrow pane the toolbar has no room for words: the mark stays, the words are for screen readers. */
const WORDS = "sr-only @2xl:not-sr-only";

/**
 * Says, in the editor toolbar, when suggestions and documentation are not
 * available and what to do about it. Quiet by design: nothing at all while
 * they work, are starting, or are turned off; a few words otherwise, with the
 * explanation and the fix one click away.
 */
export function SuggestionsStatus({
  session,
  onRetry,
}: {
  session: EditorSessionView;
  /** Starts the language server again after it failed. */
  onRetry: () => void;
}) {
  if (session.status === "restarting") {
    return (
      <span role="status" className="flex items-center gap-1.5 text-xs text-faint">
        <LoaderCircle className="size-3.5 animate-spin" aria-hidden />
        <span className={WORDS}>Restarting suggestions</span>
      </span>
    );
  }
  if (session.status !== "notInstalled" && session.status !== "failed") return null;
  const missing = session.status === "notInstalled";
  const label = missing ? "No suggestions" : "Suggestions stopped";
  return (
    <Popover>
      <PopoverTrigger
        render={
          <InlineButton
            title={label}
            className="gap-1.5 text-xs font-normal text-muted-foreground no-underline hover:text-foreground hover:no-underline"
          />
        }
      >
        <TriangleAlert className="size-3.5 text-warn" aria-hidden />
        <span className={WORDS}>{label}</span>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 gap-3 p-3">
        <PopoverHeader>
          <PopoverTitle>
            {missing ? "Suggestions are not installed" : "Suggestions stopped working"}
          </PopoverTitle>
          <PopoverDescription className="selectable text-xs text-pretty">
            {missing
              ? "Completion and documentation in the editor come from a small language server. Install it once from Settings; your code runs the same without it."
              : session.message}
          </PopoverDescription>
        </PopoverHeader>
        <div className="flex items-center gap-3 text-xs">
          {missing ? (
            <AppLink to={routes.settings("environment")}>Install in Settings</AppLink>
          ) : (
            <>
              <Button variant="outline" size="xs" onClick={onRetry}>
                Try again
              </Button>
              <AppLink to={routes.settings("environment")}>Open Settings</AppLink>
            </>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
