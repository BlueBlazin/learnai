## The idea

The policy is one predicate over three pieces of state:

$$
\text{dispatch} \iff \lnot\,\text{busy} \;\land\; |Q| > 0 \;\land\;
\bigl(|Q| \ge B \;\lor\; \text{now} \ge \text{arrival}(Q_0) + W \bigr),
$$

with $Q$ the queue, $B$ the size limit, and $W$ the wait limit. Three kinds of events
can make it true: an arrival (the queue grows), a completion (the model becomes idle),
and the passage of time (the deadline arrives). The first two are delivered to the
batcher anyway. For the third it must ask to be woken, and the only moment worth
waking for is the deadline of the oldest request, because that is the only term of the
predicate that changes with time alone.

So the whole solution is: update the state for the event, evaluate the predicate, and
report the oldest deadline as the wake-up time whenever requests wait at an idle model.
Every edge case in the statement is this rule applied to a particular order of events.
None needs a branch of its own.

Why the oldest request? The limit is a promise about latency: no request waits more
than $W$ for a batch to form. The oldest request is the one closest to breaking it, and
when it leaves, the promise passes to the next oldest, with *its* arrival time.

## Reference solution (Python)

```python
from collections import deque


class DynamicBatcher:
    def __init__(self, max_batch_size, max_wait_ms):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()   # (request_id, arrival_ms), oldest first
        self.busy = False

    def on_arrival(self, now_ms, request_id):
        self.queue.append((request_id, now_ms))
        return self._dispatch_if_due(now_ms)

    def on_batch_done(self, now_ms):
        self.busy = False
        return self._dispatch_if_due(now_ms)

    def on_timer(self, now_ms):
        return self._dispatch_if_due(now_ms)

    def next_timer_ms(self):
        if self.busy or not self.queue:
            return None
        return self._deadline()

    def _deadline(self):
        return self.queue[0][1] + self.max_wait_ms

    def _dispatch_if_due(self, now_ms):
        if self.busy or not self.queue:
            return None
        full = len(self.queue) >= self.max_batch_size
        expired = now_ms >= self._deadline()
        if not (full or expired):
            return None
        size = min(len(self.queue), self.max_batch_size)
        batch = [self.queue.popleft()[0] for _ in range(size)]
        self.busy = True
        return batch
```

The design choice that matters is that nothing about the timer is stored. A stored
deadline has to be updated at every transition (first arrival, flush with leftovers,
flush without, completion), and each of those is a place to forget. A deadline derived
from `queue[0]` is correct by construction.

The same shape works with real time. In an `asyncio` server the queue and the flag are
the same, `next_timer_ms` becomes a `loop.call_at` handle that is canceled and re-armed
after each event, and the handlers are the request coroutine, the batch task's
completion callback, and the timer callback. Keeping the policy free of I/O, as here,
is what makes it testable to the millisecond.

## Common mistakes

- **Waiting from the newest request.** Restarting the timer on every arrival (a
  "debounce") feels natural and is wrong: under a steady trickle with gaps just under
  $W$, the batch leaves only when it is full, and the first request waits $B$ times
  longer than promised.
- **Never flushing a partial batch.** Dispatching only on a full queue works in a load
  test and strands the last few requests of every burst in production.
- **Dispatching while busy.** The full-queue check and the timer both fire while a
  batch is running unless the rule checks the model first. On a real server this
  either queues a second batch inside the runtime, hiding the latency from the
  batcher, or runs two batches at once and exhausts GPU memory.
- **Off by one on the size.** `len(queue) > max_batch_size` leaves a full batch waiting
  for one more arrival or for the deadline. The batches are never too large, so only a
  check on dispatch *times* catches it.
- **Timer not re-armed after a flush.** With a stored deadline that is set "when the
  queue goes from empty to non-empty" and cleared on dispatch, the requests left behind
  by a full batch have no deadline at all. The opposite slip, not clearing it, hands
  them the deadline of a request that already left, and they are flushed early.
- **Restarting the wait at completion.** Treating the completion as the start of a new
  collection window makes a request that already queued behind a running batch wait up
  to $W$ again at an idle model.
- **Taking the newest requests.** Popping from the wrong end of the queue serves
  late arrivals first whenever a backlog exceeds one batch.
- **`>` instead of `>=` at the deadline.** The timer fires at exactly the deadline; a
  strict comparison declines to dispatch, and the timer is not due again.

## Complexity

Each handler does $O(1)$ work plus $O(k)$ to build a batch of $k$ requests, so the total
over a workload of $n$ requests is $O(n)$, with $O(|Q|)$ memory. A `deque` matters:
`list.pop(0)` would make draining a long backlog quadratic.

## In the wild

- **Triton Inference Server**: the dynamic batcher forms a batch when a preferred size
  is reached or when the oldest request has waited `max_queue_delay_microseconds`,
  the same two triggers as here, and adds priorities and per-request timeouts.
- **TorchServe** (`batchSize`, `maxBatchDelay`) and **Ray Serve** (`@serve.batch` with
  `max_batch_size`, `batch_wait_timeout_s`) expose the same two knobs.
- **LLM servers do it differently.** vLLM, TGI, and llama.cpp's server use *continuous
  batching*: generation is a loop of decoding steps, so a new request does not wait
  for the running batch to finish; it joins at the next step, and finished sequences
  leave at any step. The size limit and FIFO admission survive (as a token or memory
  budget and a waiting queue), the wait timer mostly disappears. Request-level dynamic
  batching as in this problem remains the norm for one-shot models: embeddings,
  rerankers, classifiers, vision encoders.

A useful extension is to make the wait limit adaptive: if the model is idle and
arrivals are rare, waiting only adds latency, so some systems dispatch immediately at
low load and only build batches while the model is busy, which is what happens here
naturally with `max_wait_ms = 0`.
