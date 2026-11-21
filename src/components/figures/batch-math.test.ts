import { describe, expect, it } from "vitest";
import { type BatcherCall, DynamicBatcher, EXAMPLE_ARRIVALS, exampleServiceMs, simulate } from "./batch-math";

const dispatches = (arrivals: readonly number[], maxBatchSize: number, maxWaitMs: number) =>
  simulate(arrivals, maxBatchSize, maxWaitMs).batches.map((batch) => [batch.start, batch.requests]);

describe("the example of the dynamic-batcher statement", () => {
  // max_batch_size = 3, max_wait_ms = 10, and the model takes 16 + 2n ms for a batch of n.
  const run = simulate(EXAMPLE_ARRIVALS, 3, 10);

  it("makes the calls of the table, with the same returns", () => {
    const arrival = (time: number, request: number, returned: number[] | null = null): BatcherCall => ({
      time,
      handler: "on_arrival",
      request,
      returned,
    });
    expect(run.calls).toEqual([
      arrival(0, 0),
      arrival(4, 1),
      { time: 10, handler: "on_timer", returned: [0, 1] },
      arrival(25, 2),
      arrival(26, 3),
      arrival(27, 4),
      arrival(28, 5),
      arrival(29, 6),
      { time: 30, handler: "on_batch_done", returned: [2, 3, 4] },
      { time: 52, handler: "on_batch_done", returned: [5, 6] },
      { time: 72, handler: "on_batch_done", returned: null },
      arrival(80, 7),
      arrival(84, 8),
      arrival(86, 9, [7, 8, 9]),
      { time: 108, handler: "on_batch_done", returned: null },
    ]);
  });

  it("dispatches [0, 1] at 10, [2, 3, 4] at 30, [5, 6] at 52, and [7, 8, 9] at 86", () => {
    expect(run.batches).toEqual([
      { start: 10, end: 30, requests: [0, 1] },
      { start: 30, end: 52, requests: [2, 3, 4] },
      { start: 52, end: 72, requests: [5, 6] },
      { start: 86, end: 108, requests: [7, 8, 9] },
    ]);
  });

  it("never fires the timer set for 90", () => {
    expect(run.calls.filter((call) => call.handler === "on_timer")).toHaveLength(1);
  });

  it("gives each request its wait, and the two means", () => {
    // Request 5 waits from 28 to 52: its deadline (38) passed while the model was busy.
    expect(run.waits).toEqual([10, 6, 5, 4, 3, 24, 23, 6, 2, 0]);
    expect(run.meanWait).toBeCloseTo(8.3, 12);
    expect(run.meanBatchSize).toBe(2.5);
  });

  it("asks for the wake-ups the table names", () => {
    const batcher = new DynamicBatcher(3, 10);
    expect(batcher.nextTimerMs()).toBeNull();
    expect(batcher.onArrival(0, 0)).toBeNull();
    expect(batcher.nextTimerMs()).toBe(10);
    expect(batcher.onArrival(4, 1)).toBeNull();
    expect(batcher.nextTimerMs()).toBe(10);
    expect(batcher.onTimer(10)).toEqual([0, 1]);
    expect(batcher.nextTimerMs()).toBeNull();
  });
});

describe("the cases the statement spells out", () => {
  it("includes an arrival exactly at the oldest request's deadline", () => {
    expect(dispatches([0, 10], 4, 10)).toEqual([[10, [0, 1]]]);
  });

  it("dispatches nothing while the model is busy, however many wait", () => {
    // Request 0 runs alone from 0 to 18; five more arrive meanwhile.
    expect(dispatches([0, 1, 2, 3, 4, 5], 1, 0).slice(0, 2)).toEqual([
      [0, [0]],
      [18, [1]],
    ]);
  });

  it("returns everything waiting at a completion when the oldest deadline has passed", () => {
    expect(dispatches([0, 1, 2], 1, 0)[1]).toEqual([18, [1]]);
    expect(dispatches([0, 5, 6], 3, 2)).toEqual([
      [2, [0]],
      [20, [1, 2]],
    ]);
  });

  it("does not restart a wait at a completion", () => {
    // Request 1 arrives at 25 at a busy model, which is free at 28; its deadline stays 35.
    expect(dispatches([0, 25], 3, 10)).toEqual([
      [10, [0]],
      [35, [1]],
    ]);
  });

  it("delivers a completion before an arrival at the same instant", () => {
    // The batch [0] ends at 18 with request 1 waiting past its deadline; request 2 arrives at 18 too.
    expect(dispatches([0, 1, 18], 2, 0)).toEqual([
      [0, [0]],
      [18, [1]],
      [36, [2]],
    ]);
  });

  it("treats several arrivals at one instant as separate events", () => {
    expect(dispatches([0, 0, 0, 0], 2, 50)).toEqual([
      [0, [0, 1]],
      [20, [2, 3]],
    ]);
  });

  it("dispatches a request in its own on_arrival call when max_wait_ms is 0", () => {
    expect(simulate([7], 4, 0).calls[0]).toEqual({
      time: 7,
      handler: "on_arrival",
      request: 0,
      returned: [0],
    });
  });
});

describe("every setting of the figure", () => {
  it("keeps the invariants of the policy and fits the time axis", () => {
    for (let size = 1; size <= 6; size++) {
      for (let wait = 0; wait <= 40; wait += 2) {
        const run = simulate(EXAMPLE_ARRIVALS, size, wait);
        expect(run.batches.flatMap((batch) => batch.requests)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        run.batches.forEach((batch, index) => {
          expect(batch.requests.length).toBeLessThanOrEqual(size);
          expect(batch.end - batch.start).toBe(exampleServiceMs(batch.requests.length));
          expect(batch.start).toBeGreaterThanOrEqual(run.batches[index - 1]?.end ?? 0);
          expect(batch.end).toBeLessThanOrEqual(180);
          // A partial batch never leaves early.
          const oldest = EXAMPLE_ARRIVALS[batch.requests[0] ?? 0] ?? 0;
          if (batch.requests.length < size) expect(batch.start).toBeGreaterThanOrEqual(oldest + wait);
        });
      }
    }
  });
});
