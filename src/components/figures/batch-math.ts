/**
 * The dynamic batcher behind the `batch-window` figure, exactly as
 * problems/serving/dynamic-batcher states it. After every event a batch is
 * dispatched if and only if the model is idle, the queue is not empty, and the
 * queue holds at least `max_batch_size` requests or `now_ms` has reached the
 * deadline of the oldest one (its arrival time plus `max_wait_ms`). The batch
 * is the oldest `min(max_batch_size, queue length)` requests. Requests that
 * arrive at a busy model wait, however many they are.
 */
export class DynamicBatcher {
  /** The waiting requests, oldest first. */
  private readonly queue: { id: number; arrival: number }[] = [];
  /** True from the moment a batch is returned until `onBatchDone`. */
  private busy = false;

  constructor(
    private readonly maxBatchSize: number,
    private readonly maxWaitMs: number,
  ) {}

  onArrival(nowMs: number, requestId: number): number[] | null {
    this.queue.push({ id: requestId, arrival: nowMs });
    return this.dispatchIfDue(nowMs);
  }

  onBatchDone(nowMs: number): number[] | null {
    this.busy = false;
    return this.dispatchIfDue(nowMs);
  }

  onTimer(nowMs: number): number[] | null {
    return this.dispatchIfDue(nowMs);
  }

  /** The deadline of the oldest request while requests wait at an idle model; otherwise no wake-up. */
  nextTimerMs(): number | null {
    const oldest = this.queue[0];
    return this.busy || !oldest ? null : oldest.arrival + this.maxWaitMs;
  }

  private dispatchIfDue(nowMs: number): number[] | null {
    const oldest = this.queue[0];
    if (this.busy || !oldest) return null;
    const full = this.queue.length >= this.maxBatchSize;
    const expired = nowMs >= oldest.arrival + this.maxWaitMs;
    if (!full && !expired) return null;
    this.busy = true;
    return this.queue.splice(0, this.maxBatchSize).map((request) => request.id);
  }
}

/** One handler call of a simulated run, as a row of the statement's example table. */
export interface BatcherCall {
  time: number;
  handler: "on_arrival" | "on_batch_done" | "on_timer";
  /** The request of an `on_arrival` call. */
  request?: number;
  returned: number[] | null;
}

export interface Batch {
  /** When the batch was dispatched and when the model finished it. */
  start: number;
  end: number;
  /** The requests of the batch, oldest first. */
  requests: number[];
}

export interface BatcherRun {
  calls: BatcherCall[];
  batches: Batch[];
  /** For each request, by index: the time from its arrival to its dispatch. */
  waits: number[];
  meanWait: number;
  meanBatchSize: number;
}

/** How long the model of the statement's example runs a batch of `n` requests: `16 + 2n` ms. */
export const exampleServiceMs = (n: number) => 16 + 2 * n;

/** The arrival times of requests 0 to 9 in the statement's example. */
export const EXAMPLE_ARRIVALS: readonly number[] = [0, 4, 25, 26, 27, 28, 29, 80, 84, 86];

/**
 * Drives a batcher over `arrivals` (times in ms, sorted; request `i` arrives at
 * `arrivals[i]`) the way the statement's simulator does. Events at one instant
 * are delivered in the order batch completion, arrivals, timer; after every
 * call the batcher is asked for its next wake-up, and an answer that is not
 * later than the current time is dropped.
 */
export function simulate(
  arrivals: readonly number[],
  maxBatchSize: number,
  maxWaitMs: number,
  serviceMs: (n: number) => number = exampleServiceMs,
): BatcherRun {
  const batcher = new DynamicBatcher(maxBatchSize, maxWaitMs);
  const calls: BatcherCall[] = [];
  const batches: Batch[] = [];
  const waits = new Array<number>(arrivals.length).fill(0);
  let nextArrival = 0;
  let doneAt: number | null = null;
  let timerAt: number | null = null;

  for (;;) {
    // The earliest event; at equal times the completion is first and the timer last.
    const arrivalAt = arrivals[nextArrival] ?? null;
    const due = [doneAt, arrivalAt, timerAt].filter((time) => time !== null);
    if (due.length === 0) break;
    const now = Math.min(...due);

    let call: BatcherCall;
    if (doneAt === now) {
      doneAt = null;
      call = { time: now, handler: "on_batch_done", returned: batcher.onBatchDone(now) };
    } else if (arrivalAt === now) {
      const request = nextArrival++;
      call = { time: now, handler: "on_arrival", request, returned: batcher.onArrival(now, request) };
    } else {
      call = { time: now, handler: "on_timer", returned: batcher.onTimer(now) };
    }
    calls.push(call);

    if (call.returned) {
      doneAt = now + serviceMs(call.returned.length);
      batches.push({ start: now, end: doneAt, requests: call.returned });
      for (const request of call.returned) waits[request] = now - (arrivals[request] ?? now);
    }
    const answer = batcher.nextTimerMs();
    timerAt = answer !== null && answer > now ? answer : null;
  }

  const mean = (values: readonly number[]) =>
    values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    calls,
    batches,
    waits,
    meanWait: mean(waits),
    meanBatchSize: mean(batches.map((batch) => batch.requests.length)),
  };
}
