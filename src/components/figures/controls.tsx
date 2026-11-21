import { cn } from "cn";
import { type ReactNode, useId } from "react";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

/** A slider of an interactive figure: its name and its value on one line, the slider beneath. */
export function FigureSlider({
  name,
  value,
  shown,
  min,
  max,
  step,
  onChange,
}: {
  name: string;
  value: number;
  /** The value as text, for example with "(off)" where the value switches a stage off. */
  shown: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <span id={id} className="font-mono text-xs text-muted-foreground">
          {name}
        </span>
        <span className="font-mono text-xs text-foreground tabular-nums">{shown}</span>
      </div>
      <Slider
        aria-labelledby={id}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => {
          const first = Array.isArray(next) ? next[0] : next;
          // Steps of 0.05 and 0.1 are not exact in floating point; two decimals are.
          if (typeof first === "number") onChange(Math.round(first * 100) / 100);
        }}
      />
    </div>
  );
}

const SLIDER_COLUMNS = { 2: "grid-cols-2", 3: "grid-cols-3" } as const;

/** The sliders of an interactive figure, side by side beneath the drawing. */
export function FigureSliders({ columns = 3, children }: { columns?: 2 | 3; children: ReactNode }) {
  return <div className={cn("mt-4 grid gap-x-6 gap-y-3 font-sans", SLIDER_COLUMNS[columns])}>{children}</div>;
}

/** A choice between a few named options of an interactive figure: exactly one is always on. */
export function FigureChoice<Value extends string>({
  name,
  label,
  value,
  options,
  onChange,
}: {
  /** The word set before the options. */
  name: string;
  /** The accessible name of the group. */
  label: string;
  value: Value;
  options: readonly { value: Value; label: string }[];
  onChange: (value: Value) => void;
}) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-3 font-sans text-sm text-muted-foreground">
      <span>{name}</span>
      <ToggleGroup
        variant="outline"
        size="sm"
        spacing={0}
        aria-label={label}
        value={[value]}
        onValueChange={(next) => {
          const choice = (next as Value[])[0];
          if (choice) onChange(choice);
        }}
      >
        {options.map((option) => (
          <ToggleGroupItem
            key={option.value}
            value={option.value}
            className="text-muted-foreground aria-pressed:bg-accent aria-pressed:text-foreground"
          >
            {option.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}

/** The one line beneath the controls of an interactive figure that says what the drawing shows. */
export function FigureReadout({ children }: { children: ReactNode }) {
  return <div className="mt-3 font-sans text-base text-foreground">{children}</div>;
}

/** The readout as named numbers on one line: `FigureNumber` children. */
export function FigureNumbers({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3 flex flex-wrap items-baseline gap-x-5 gap-y-1 font-sans text-base">{children}</div>
  );
}

/** One named number of a readout. */
export function FigureNumber({ name, children }: { name: ReactNode; children: ReactNode }) {
  return (
    <span className="flex items-baseline gap-2">
      <span className="text-sm text-muted-foreground">{name}</span>
      <span className="font-mono text-sm text-foreground tabular-nums">{children}</span>
    </span>
  );
}
