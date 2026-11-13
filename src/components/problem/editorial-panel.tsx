import { Lock } from "lucide-react";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { CodeBlock } from "@/components/prose/code-block";
import { Prose } from "@/components/prose/prose";
import { Button } from "@/components/ui/button";
import type { Framework } from "@/lib/api";
import { FRAMEWORK_LABEL } from "@/lib/labels";

/**
 * The editorial and reference solution. Locked until the problem is solved;
 * revealing it early is allowed but takes a deliberate confirmation.
 */
export function EditorialPanel({
  unlocked,
  editorial,
  reference,
  referenceError,
  framework,
  revealing = false,
  onReveal,
}: {
  unlocked: boolean;
  editorial: string | null;
  /** The reference solution for `framework`; undefined while loading. */
  reference?: string;
  referenceError?: unknown;
  framework: Framework | null;
  revealing?: boolean;
  onReveal: () => void;
}) {
  if (!unlocked) {
    return (
      <div className="flex max-w-measure flex-col items-start gap-2">
        <p className="flex items-center gap-2 font-serif text-md">
          <Lock className="size-3.5 text-muted-foreground" aria-hidden />
          The editorial is locked
        </p>
        <p className="text-sm text-muted-foreground">
          It opens when you solve the problem, with the reference solution beside it. You can reveal it now,
          but the hints will get you further for less.
        </p>
        <ConfirmDialog
          trigger={
            <Button variant="outline" size="sm" className="mt-2" disabled={revealing}>
              Reveal editorial
            </Button>
          }
          title="Reveal the editorial?"
          description="You will see the full explanation and the reference solution. This cannot be undone for this problem."
          confirmLabel="Reveal editorial"
          onConfirm={onReveal}
        />
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-8">
      <Prose>{editorial ?? ""}</Prose>
      {framework ? (
        <section className="max-w-measure border-t border-rule pt-3">
          <h2 className="pb-3 text-sm font-semibold">Reference solution, {FRAMEWORK_LABEL[framework]}</h2>
          {referenceError ? (
            <ErrorState title="The reference could not be loaded" error={referenceError} className="py-0" />
          ) : reference === undefined ? (
            <LoadingLines lines={5} />
          ) : (
            <CodeBlock code={reference} language="python" lineNumbers />
          )}
        </section>
      ) : null}
    </div>
  );
}
