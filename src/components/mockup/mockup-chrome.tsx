import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { MockState, MockStateId } from "./mock-data";

/**
 * Mockup chrome, not part of the proposed design: picks which sample state the
 * page below shows. Dashed lines mark everything that belongs to the mockup
 * itself; the design only ever uses solid rules.
 */
export function StateSwitcher({
  states,
  value,
  onChange,
}: {
  states: MockState[];
  value: MockStateId;
  onChange: (value: MockStateId) => void;
}) {
  return (
    <div className="mb-10 flex items-center gap-4 border-b border-dashed border-input pb-3 text-xs text-muted-foreground">
      <span>Mockup controls: sample state</span>
      <ToggleGroup
        variant="outline"
        size="sm"
        spacing={0}
        aria-label="Sample state"
        value={[value]}
        onValueChange={(next) => {
          const choice = (next as MockStateId[])[0];
          if (choice) onChange(choice);
        }}
      >
        {states.map((state) => (
          <ToggleGroupItem key={state.id} value={state.id}>
            {state.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <span className="ml-auto">Everything below the dashed line is the proposal.</span>
    </div>
  );
}

/** Mockup chrome: a dashed line and a note about the proposal, at the foot of the page. */
export function MockupNote({ children }: { children: string }) {
  return (
    <p className="mt-2 border-t border-dashed border-input pt-3 text-xs text-muted-foreground">{children}</p>
  );
}
