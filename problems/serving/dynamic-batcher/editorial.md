## The idea

The entire policy reduces to one predicate over three pieces of state: the waiting queue $Q$ (ordered oldest first), a boolean flag `busy` indicating whether the model is running, and the current time $\text{now}$. Let $B$ be `max_batch_size`, $W$ be `max_wait_ms`, and $\text{arrival}(Q_0)$ be the arrival time of the oldest request in $Q$. A batch is dispatched if and only if:

$$
\text{dispatch} \iff \lnot\,\text{busy} \;\land\; |Q| > 0 \;\land\;
\bigl(|Q| \ge B \;\lor\; \text{now} \ge \text{arrival}(Q_0) + W \bigr)
$$

Three events can make this predicate true: an arrival grows the queue, a batch completion makes the model idle, or time advances to the oldest request's deadline. The simulator delivers arrivals and completions automatically. For time advancing, the batcher must request a wake-up. The only timestamp worth waking for is $\text{arrival}(Q_0) + W$, because it is the only term in the predicate that changes with time alone.

Every handler updates state for its event, evaluates the dispatch predicate, and relies on `next_timer_ms()` to report the oldest deadline whenever requests are waiting at an idle model. Every edge case in the specification falls out of this single rule without special-case branches.

Using the oldest request's arrival time enforces the latency guarantee. No request may wait more than $W$ milliseconds at an idle model for a batch to form. The oldest request is always closest to violating that bound, and once it is dispatched, the bound transfers to the next oldest request using its own arrival time.

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

The key design choice is not storing any timer state. A stored deadline must be updated on every transition (first arrival, flush with leftovers, flush without leftovers, and batch completion), and missing any one of them introduces a bug. Deriving the deadline directly from `queue[0]` keeps it consistent automatically.

The same structure works with real time. In an `asyncio` server, the queue and `busy` flag stay identical, `next_timer_ms()` maps to a `loop.call_at` handle that is canceled and re-armed after each event, and the handlers become the request coroutine, the batch completion callback, and the timer callback. Keeping the policy separate from I/O makes it deterministic and testable to the millisecond.

## Common mistakes

- **Waiting from the newest request.** Restarting the timer on every arrival (a debounce) breaks the latency bound. Under a steady trickle of requests spaced just under $W$ apart, the batch only leaves when full, making the first request wait up to $B$ times longer than promised.
- **Never flushing a partial batch.** Dispatching only when the queue reaches `max_batch_size` passes high-load tests but strands the trailing requests of every burst in production.
- **Dispatching while busy.** Both the full-queue check and the timer can fire while a batch is running if you forget to check `self.busy` first. On a real server, this either queues hidden work inside the runtime or launches two concurrent batches and exhausts GPU memory.
- **Off by one on batch size.** Writing `len(queue) > max_batch_size` leaves a full batch waiting for one extra arrival or for its deadline. Because no batch ever exceeds `max_batch_size`, only checking dispatch timestamps catches this bug.
- **Timer not re-armed after a flush.** If you store a deadline when the queue becomes non-empty and clear it on dispatch, leftover requests after a full batch have no timer. Conversely, failing to clear a stored deadline applies the old request's deadline to the leftovers and flushes them early.
- **Restarting the wait at completion.** Starting a fresh collection window when `on_batch_done` fires forces requests that already waited behind a running batch to wait up to another $W$ milliseconds at an idle model.
- **Taking the newest requests.** Popping from the right end of the queue serves late arrivals before older ones whenever a backlog exceeds one batch.
- **Using `>` instead of `>=` at the deadline.** The timer fires at `now_ms == deadline`. A strict `>` check declines to dispatch, and the timer will not fire again.

## Complexity

Each handler performs $O(1)$ work plus $O(k)$ to pop and return a batch of $k$ requests. Over a workload of $n$ requests, total runtime is $O(n)$ and memory is $O(|Q|)$. Using a `deque` avoids the quadratic cost of `list.pop(0)` when draining a long backlog.

## In the wild

- **Triton Inference Server** forms a batch when a preferred size is reached or when the oldest request has waited `max_queue_delay_microseconds`. It uses the same two triggers and adds priority queues and per-request timeouts.
- **TorchServe** (`batchSize`, `maxBatchDelay`) and **Ray Serve** (`@serve.batch` with `max_batch_size`, `batch_wait_timeout_s`) expose the same two controls.
- **LLM servers use continuous batching.** Systems like vLLM, TGI, and `llama.cpp` run generation as a loop of decode steps. New requests join at the next step instead of waiting for the current batch to finish, and completed sequences exit immediately. The size limit and FIFO waiting queue remain (often expressed as a token or KV-cache budget), while the wait timer mostly disappears. Request-level dynamic batching as implemented here remains standard for one-shot models such as embeddings, rerankers, classifiers, and vision encoders.

Some systems also use an adaptive wait limit. When the model is idle and traffic is sparse, waiting only adds latency, so they dispatch immediately at low load and form batches only while the model is busy. This policy already behaves that way when `max_wait_ms = 0`.
