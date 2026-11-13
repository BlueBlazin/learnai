import { Lock } from "lucide-react";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
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
      <EmptyState
        className="py-0"
        icon={<Lock className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />}
        title="The editorial is locked"
        action={
          <ConfirmDialog
            trigger={
              <Button variant="outline" size="sm" disabled={revealing}>
                Reveal editorial
              </Button>
            }
            title="Reveal the editorial?"
            description="You will see the full explanation and the reference solution. This cannot be undone for this problem."
            confirmLabel="Reveal editorial"
            onConfirm={onReveal}
          />
        }
      >
        It opens when you solve the problem, with the reference solution beside it. You can reveal it now, but
        the hints will get you further for less.
      </EmptyState>
    );
  }
  return (
    <div className="flex flex-col gap-8">
      <Prose>{editorial ?? ""}</Prose>
      {framework ? (
        <Section title={`Reference solution, ${FRAMEWORK_LABEL[framework]}`} className="max-w-measure pb-0">
          {referenceError ? (
            <ErrorState title="The reference could not be loaded" error={referenceError} className="py-0" />
          ) : reference === undefined ? (
            <LoadingLines lines={5} />
          ) : (
            <CodeBlock code={reference} language="python" lineNumbers />
          )}
        </Section>
      ) : null}
    </div>
  );
}
