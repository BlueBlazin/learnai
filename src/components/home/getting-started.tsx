import type { ReactNode } from "react";
import { SHORTCUTS, ShortcutHint } from "@/components/app/shortcut-hint";

const STEPS: { title: string; body: ReactNode }[] = [
  {
    title: "Read the statement",
    body: "Every problem gives an interface, its constraints, and a worked example. Hints come one at a time, and only when you ask.",
  },
  {
    title: "Write the solution",
    body: (
      <>
        In the editor beside the statement, or in your own: each solution is a plain file under{" "}
        <code className="font-mono text-xs">workspace/</code>, one per framework.
      </>
    ),
  },
  {
    title: "Run, then submit",
    body: (
      <>
        <p>
          Run checks the public cases. Submit runs every case, hidden ones included, and records the attempt.
        </p>
        <p className="mt-1.5 flex gap-5">
          <ShortcutHint label="Run" keys={SHORTCUTS.run} />
          <ShortcutHint label="Submit" keys={SHORTCUTS.submit} />
        </p>
      </>
    ),
  },
  {
    title: "Compare with the editorial",
    body: "An accepted submission opens the explanation and the reference solution.",
  },
];

/**
 * How one problem goes, start to finish. Shown on Home until there is work to
 * report, in place of a progress summary that would be all zeros.
 */
export function GettingStarted() {
  return (
    <ol className="flex flex-col gap-4">
      {STEPS.map((step, index) => (
        <li key={step.title} className="flex items-baseline gap-3">
          <span className="tabular w-3 shrink-0 text-sm font-medium text-muted-foreground" aria-hidden>
            {index + 1}
          </span>
          <div className="min-w-0">
            <p className="font-serif text-md">{step.title}</p>
            <div className="mt-0.5 text-sm text-pretty text-muted-foreground">{step.body}</div>
          </div>
        </li>
      ))}
    </ol>
  );
}
