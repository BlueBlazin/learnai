import { useId, useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { InteractiveFigureProps } from "./registry";
import { keptCount, runSampler, type StageOrder, softmax } from "./sampler-math";

/** Ten candidate tokens and the model's logits for them, most likely first. */
const TOKENS = ["the", "a", "his", "one", "that", "my", "an", "this", "her", "two"] as const;
const LOGITS = [3.2, 2.9, 2.5, 2.2, 1.8, 1.4, 0.9, 0.4, -0.2, -0.9] as const;
const MODEL = softmax(LOGITS);

const ORDERS: readonly { value: StageOrder; label: string }[] = [
  { value: "filters-first", label: "top-k, top-p, temperature" },
  { value: "temperature-first", label: "temperature, top-k, top-p" },
];

/** The chart, in drawing units: a 480-unit drawing like an author's figure. */
const CHART = { width: 480, height: 204, left: 40, right: 474, base: 156, rise: 136, slot: 43.5, bar: 28 };
const TICKS = [0, 0.25, 0.5, 0.75, 1];

/** A probability the way the bars label it: two decimals, no leading zero. */
const short = (probability: number) => probability.toFixed(2).replace(/^0/, "");

function Knob({
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
  /** The value as text, with "(off)" where the value switches the stage off. */
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

/**
 * `sampler-order`: ten tokens under top-k, top-p, and temperature, with the
 * temperature after the filters or before them. The dashed outline is the
 * model's own distribution; a filled bar is the probability the chain samples
 * from. The readout says how many tokens each order keeps.
 */
export default function SamplerOrder({ label }: InteractiveFigureProps) {
  const [topK, setTopK] = useState(8);
  const [topP, setTopP] = useState(0.9);
  const [temperature, setTemperature] = useState(2);
  const [order, setOrder] = useState<StageOrder>("filters-first");

  const result = runSampler(LOGITS, { topK, topP, temperature, order });
  const other = runSampler(LOGITS, {
    topK,
    topP,
    temperature,
    order: order === "filters-first" ? "temperature-first" : "filters-first",
  });
  const kept = keptCount(result);
  const otherKept = keptCount(other);
  const names = TOKENS.filter((_, token) => result.removedBy[token] === null);

  return (
    <>
      <FigureDrawing label={label}>
        {/* The host is the image and carries the name; the readout below says what the bars show. */}
        <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="figure-bars" aria-hidden="true">
          {TICKS.map((tick) => {
            const y = CHART.base - tick * CHART.rise;
            return (
              <g key={tick}>
                {tick > 0 ? <line className="grid" x1={CHART.left} y1={y} x2={CHART.right} y2={y} /> : null}
                <text className="tick" x={CHART.left - 6} y={y} textAnchor="end" dominantBaseline="central">
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}
          {TOKENS.map((name, token) => {
            const x = CHART.left + 8 + token * CHART.slot;
            const middle = x + CHART.bar / 2;
            const modelHeight = Math.max((MODEL[token] ?? 0) * CHART.rise, 1);
            const probability = result.probabilities[token] ?? 0;
            const height = Math.max(probability * CHART.rise, 1);
            const removedBy = result.removedBy[token] ?? null;
            return (
              <g key={name}>
                <rect
                  className="model"
                  x={x}
                  y={CHART.base - modelHeight}
                  width={CHART.bar}
                  height={modelHeight}
                />
                {removedBy === null ? (
                  <>
                    <rect
                      className="kept"
                      x={x}
                      y={CHART.base - height}
                      width={CHART.bar}
                      height={height}
                      rx={2}
                    />
                    <text
                      className="value"
                      x={middle}
                      y={CHART.base - Math.max(height, modelHeight) - 8}
                      textAnchor="middle"
                      dominantBaseline="central"
                    >
                      {short(probability)}
                    </text>
                  </>
                ) : (
                  <text
                    className="removed-by"
                    x={middle}
                    y={CHART.base + 31}
                    textAnchor="middle"
                    dominantBaseline="central"
                  >
                    {removedBy}
                  </text>
                )}
                <text
                  className={removedBy === null ? "token" : "token removed"}
                  x={middle}
                  y={CHART.base + 15}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  {name}
                </text>
              </g>
            );
          })}
          <line className="axis" x1={CHART.left} y1={CHART.base} x2={CHART.right} y2={CHART.base} />
        </svg>
      </FigureDrawing>
      <div className="mt-4 grid grid-cols-3 gap-x-6 gap-y-3 font-sans">
        <Knob
          name="top_k"
          value={topK}
          shown={topK >= TOKENS.length ? `${topK} (off)` : String(topK)}
          min={1}
          max={TOKENS.length}
          step={1}
          onChange={setTopK}
        />
        <Knob
          name="top_p"
          value={topP}
          shown={topP >= 1 ? "1.00 (off)" : topP.toFixed(2)}
          min={0.05}
          max={1}
          step={0.05}
          onChange={setTopP}
        />
        <Knob
          name="temperature"
          value={temperature}
          shown={temperature.toFixed(1)}
          min={0.2}
          max={2}
          step={0.1}
          onChange={setTemperature}
        />
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3 font-sans text-sm text-muted-foreground">
        <span>Order</span>
        <ToggleGroup
          variant="outline"
          size="sm"
          spacing={0}
          aria-label="Order of the stages"
          value={[order]}
          onValueChange={(next) => {
            const choice = (next as StageOrder[])[0];
            if (choice) setOrder(choice);
          }}
        >
          {ORDERS.map(({ value, label: text }) => (
            <ToggleGroupItem
              key={value}
              value={value}
              className="text-muted-foreground aria-pressed:bg-accent aria-pressed:text-foreground"
            >
              {text}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      <div className="mt-3 font-sans text-base text-foreground">
        Kept {kept} of {TOKENS.length}: {names.join(", ")}.{" "}
        {otherKept === kept ? "The other order keeps the same set." : `The other order keeps ${otherKept}.`}
      </div>
    </>
  );
}
