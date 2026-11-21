import { useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { FigureChoice, FigureReadout, FigureSlider, FigureSliders } from "./controls";
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
      <FigureSliders>
        <FigureSlider
          name="top_k"
          value={topK}
          shown={topK >= TOKENS.length ? `${topK} (off)` : String(topK)}
          min={1}
          max={TOKENS.length}
          step={1}
          onChange={setTopK}
        />
        <FigureSlider
          name="top_p"
          value={topP}
          shown={topP >= 1 ? "1.00 (off)" : topP.toFixed(2)}
          min={0.05}
          max={1}
          step={0.05}
          onChange={setTopP}
        />
        <FigureSlider
          name="temperature"
          value={temperature}
          shown={temperature.toFixed(1)}
          min={0.2}
          max={2}
          step={0.1}
          onChange={setTemperature}
        />
      </FigureSliders>
      <FigureChoice
        name="Order"
        label="Order of the stages"
        value={order}
        options={ORDERS}
        onChange={setOrder}
      />
      <FigureReadout>
        Kept {kept} of {TOKENS.length}: {names.join(", ")}.{" "}
        {otherKept === kept ? "The other order keeps the same set." : `The other order keeps ${otherKept}.`}
      </FigureReadout>
    </>
  );
}
