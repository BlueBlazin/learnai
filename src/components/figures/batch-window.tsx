import { useState } from "react";
import { FigureDrawing } from "@/components/prose/figure";
import { EXAMPLE_ARRIVALS, simulate } from "./batch-math";
import { FigureNumber, FigureNumbers, FigureSlider, FigureSliders } from "./controls";
import type { InteractiveFigureProps } from "./registry";

/** The timeline, in drawing units: a 480-unit drawing like an author's figure. */
const CHART = { width: 480, height: 190, left: 40, right: 440, top: 22, row: 14, end: 180 };
const BASE = CHART.top + EXAMPLE_ARRIVALS.length * CHART.row + 4;
const TICKS = [0, 30, 60, 90, 120, 150, 180];

const xOf = (ms: number) => CHART.left + (ms / CHART.end) * (CHART.right - CHART.left);
const middleOf = (request: number) => CHART.top + (request + 0.5) * CHART.row;

/**
 * `batch-window`: the ten arrivals of the example in
 * problems/serving/dynamic-batcher (a batch of n runs for 16 + 2n ms) under a
 * chosen `max_batch_size` and `max_wait_ms`. A row is a request: a dot at its
 * arrival and a line while it waits. A box is a batch while the model runs
 * it, over the rows of its requests. A tick on a waiting line is a deadline
 * that passed while the model was busy. The column on the right is each
 * request's wait.
 */
export default function BatchWindow({ label }: InteractiveFigureProps) {
  const [maxBatchSize, setMaxBatchSize] = useState(3);
  const [maxWaitMs, setMaxWaitMs] = useState(10);

  const run = simulate(EXAMPLE_ARRIVALS, maxBatchSize, maxWaitMs);

  return (
    <>
      <FigureDrawing label={label}>
        {/* The host is the image and carries the name; the readout below gives the numbers. */}
        <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="figure-timeline" aria-hidden="true">
          <text className="head" x={CHART.left - 10} y={10} textAnchor="end" dominantBaseline="central">
            request
          </text>
          <text className="head" x={CHART.width - 4} y={10} textAnchor="end" dominantBaseline="central">
            wait, ms
          </text>
          {TICKS.map((tick) => (
            <g key={tick}>
              <line className="grid" x1={xOf(tick)} y1={CHART.top} x2={xOf(tick)} y2={BASE} />
              <text
                className="tick"
                x={xOf(tick)}
                y={BASE + 14}
                textAnchor="middle"
                dominantBaseline="central"
              >
                {tick}
              </text>
            </g>
          ))}
          <text
            className="head"
            x={CHART.width - 4}
            y={BASE + 14}
            textAnchor="end"
            dominantBaseline="central"
          >
            ms
          </text>
          <line className="axis" x1={CHART.left} y1={BASE} x2={CHART.right} y2={BASE} />
          {run.batches.map((batch) => {
            const first = batch.requests[0] ?? 0;
            const height = batch.requests.length * CHART.row - 3;
            return (
              <g key={batch.start}>
                <rect
                  className="batch"
                  x={xOf(batch.start)}
                  y={CHART.top + first * CHART.row + 1.5}
                  width={xOf(batch.end) - xOf(batch.start)}
                  height={height}
                  rx={2}
                />
                <text
                  className="size"
                  x={(xOf(batch.start) + xOf(batch.end)) / 2}
                  y={CHART.top + first * CHART.row + 1.5 + height / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                >
                  n = {batch.requests.length}
                </text>
                {batch.requests.map((request) => {
                  const arrival = EXAMPLE_ARRIVALS[request] ?? 0;
                  const deadline = arrival + maxWaitMs;
                  const y = middleOf(request);
                  return (
                    <g key={request}>
                      <line className="wait" x1={xOf(arrival)} y1={y} x2={xOf(batch.start)} y2={y} />
                      {maxWaitMs > 0 && deadline < batch.start ? (
                        <line
                          className="deadline"
                          x1={xOf(deadline)}
                          y1={y - 4}
                          x2={xOf(deadline)}
                          y2={y + 4}
                        />
                      ) : null}
                      <circle className="arrival" cx={xOf(arrival)} cy={y} r={2.2} />
                    </g>
                  );
                })}
              </g>
            );
          })}
          {run.waits.map((wait, request) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the requests are fixed and never reorder.
            <g key={request}>
              <text
                className="id"
                x={CHART.left - 10}
                y={middleOf(request)}
                textAnchor="end"
                dominantBaseline="central"
              >
                {request}
              </text>
              <text
                className="value"
                x={CHART.width - 4}
                y={middleOf(request)}
                textAnchor="end"
                dominantBaseline="central"
              >
                {wait}
              </text>
            </g>
          ))}
        </svg>
      </FigureDrawing>
      <FigureSliders columns={2}>
        <FigureSlider
          name="max_batch_size"
          value={maxBatchSize}
          shown={String(maxBatchSize)}
          min={1}
          max={6}
          step={1}
          onChange={setMaxBatchSize}
        />
        <FigureSlider
          name="max_wait_ms"
          value={maxWaitMs}
          shown={String(maxWaitMs)}
          min={0}
          max={40}
          step={2}
          onChange={setMaxWaitMs}
        />
      </FigureSliders>
      <FigureNumbers>
        <FigureNumber name="Batches">{run.batches.length}</FigureNumber>
        <FigureNumber name="Mean batch size">{run.meanBatchSize.toFixed(2)}</FigureNumber>
        <FigureNumber name="Mean wait">{run.meanWait.toFixed(1)} ms</FigureNumber>
      </FigureNumbers>
    </>
  );
}
