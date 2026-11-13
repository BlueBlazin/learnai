import { RuledList } from "@/components/app/ruled-list";
import { EmptyState } from "@/components/app/states";
import { Prose } from "@/components/prose/prose";
import { Button } from "@/components/ui/button";

/**
 * Hints, revealed one at a time and in order. Each hint gives away a little
 * more than the last, so the next one stays hidden until asked for.
 */
export function HintList({
  hints,
  revealed,
  onReveal,
}: {
  hints: string[];
  /** How many hints are currently shown. */
  revealed: number;
  onReveal: (count: number) => void;
}) {
  if (hints.length === 0) {
    return <EmptyState title="This problem has no hints">The statement is all you get.</EmptyState>;
  }
  const shown = Math.min(revealed, hints.length);
  return (
    <div className="max-w-measure">
      <RuledList ordered framed={false} className="border-b border-border/70">
        {hints.slice(0, shown).map((hint, index) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: hints are an ordered, static list.
          <li key={index} className="py-5 first:pt-0">
            <p className="pb-1.5 text-xs font-semibold text-muted-foreground">
              Hint {index + 1} of {hints.length}
            </p>
            <Prose size="compact">{hint}</Prose>
          </li>
        ))}
      </RuledList>
      {shown < hints.length ? (
        <div className="flex flex-col items-start gap-2 pt-5">
          <p className="text-sm text-muted-foreground">
            {shown === 0
              ? `There are ${hints.length} hints, from a nudge to nearly the answer. Take them one at a time.`
              : "The next hint gives away more."}
          </p>
          <Button variant="outline" size="sm" onClick={() => onReveal(shown + 1)}>
            Show hint {shown + 1}
          </Button>
        </div>
      ) : (
        <p className="pt-5 text-sm text-muted-foreground">That was the last hint.</p>
      )}
    </div>
  );
}
