import type { ReactNode } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MOCKUP_VIEWS, type MockupView } from "./model";

const VIEW_LABEL: Record<MockupView, string> = {
  areas: "Areas",
  area: "Area",
  module: "Module",
};

/**
 * Mockup chrome, not part of the design: the three views as text tabs over a
 * dashed rule, with a note beside them.
 */
export function MockupSwitcher({
  view,
  onChange,
  note,
}: {
  view: MockupView;
  onChange: (view: MockupView) => void;
  note?: ReactNode;
}) {
  return (
    <div className="mb-10 flex items-center gap-5 border-b border-dashed text-xs text-muted-foreground">
      <span className="font-semibold text-foreground">Mockup</span>
      <Tabs value={view} onValueChange={(value) => onChange(value as MockupView)}>
        <TabsList aria-label="Mockup views" className="h-toolbar">
          {MOCKUP_VIEWS.map((candidate) => (
            <TabsTrigger key={candidate} value={candidate}>
              {VIEW_LABEL[candidate]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      {note ? <span className="ml-auto text-right text-pretty">{note}</span> : null}
    </div>
  );
}
