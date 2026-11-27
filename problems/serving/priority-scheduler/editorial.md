## The idea

Strict priority scheduling always serves the most urgent request, which starves lower-priority work whenever urgent requests keep arriving. Aging fixes this by increasing a request's urgency as it waits. At virtual time $t$, a request's priority improves by one level for every full `aging_ms` elapsed since its arrival:

$$
\text{effective}(t) = \text{priority} - \left\lfloor \frac{t - \text{arrival}}{\text{aging\_ms}} \right\rfloor
$$

A request with priority $p$ reaches the level of a newly arrived priority-0 request after waiting $p \cdot \text{aging\_ms}$. Once it reaches that level, it beats every request that arrived after it because ties go to the earlier arrival. Its extra wait compared to a priority-0 request arriving at the same instant is bounded by roughly $(p + 1) \cdot \text{aging\_ms}$, turning starvation into a predictable delay.

Because a request's rank depends on the current time, two waiting requests with different arrival times cross their aging boundaries at different instants. Their relative order can flip while nothing arrives and nothing is admitted. Any sort key stored at insertion becomes stale as soon as the clock advances. Evaluating the selection rule at admission time by scanning `waiting` with `now_ms` avoids stale keys completely.

The tie-breaking rules also matter. Breaking ties by `arrival_s` before queue position ensures that a request arriving mid-step is not penalized relative to when the scheduler drains the queue. Ignoring `request.id` keeps ids opaque.

## Reference solution (Python)

```python
def effective_priority(request, now_ms, aging_ms):
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)


def pick_next(waiting, now_ms, aging_ms):
    best = min(waiting, key=lambda r: (effective_priority(r, now_ms, aging_ms), r.arrival_s))
    waiting.remove(best)        # `min` returned the first of equal keys: queue order
    return best


async def run(engine, queue, max_running, aging_ms):
    loop = asyncio.get_running_loop()
    waiting, running = [], []
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        now_ms = loop.time()
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = pick_next(waiting, now_ms, aging_ms)
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)
        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
```

The loop is identical to `continuous-batching-scheduler` except for the admission call. In an iteration-level serving engine, the loop manages execution steps while the scheduling policy simply selects which waiting request gets the next free slot.

## Common mistakes

- **No aging.** Returning the base priority passes static ordering tests but starves low-priority requests under a steady urgent stream (4946 ms instead of 1320 ms on the stress workload, and unbounded on longer streams).
- **A static sort key.** A heap keyed by `(priority, arrival)` at insertion cannot represent aging. Even if you rebuild keys on every arrival, the order is still stale when a slot opens with no arrivals in between.
- **Fractional aging.** Computing `priority - waited / aging_ms` is a different policy. Under fractional aging, a priority-1 request that has waited 90% of a period still ranks behind a fresh priority-0 request, whereas two requests with different remainders break ties differently than whole-level aging. It also collapses to a static key (`priority + arrival / aging_ms`), which is why it is a common mistake.
- **Off by one at the boundary.** A request gains a level at exactly `aging_ms`, which requires `floor`, not `ceil - 1`. Because arrival times and step durations lie on a 0.25 ms grid, exact boundaries happen frequently.
- **Measuring wait time from collection.** Recording `loop.time()` when draining `queue` drops the time a request waited during the previous step. Always measure from `request.arrival_s`.
- **Breaking ties by id.** Sorting by `(effective, id)` passes hand-written tests where ids increase with arrival time, but fails when ids are arbitrary strings or out-of-order integers.
- **Larger number means more urgent.** Following vLLM and Unix `nice`, smaller priority values are more urgent here.

## Complexity

`pick_next` performs an $O(W)$ linear scan over $W$ waiting requests per admission. That cost is negligible next to a forward pass of several milliseconds, and it handles time-varying keys cleanly. If $W$ grew to hundreds of thousands, you could keep one FIFO queue per base priority level. Within a single base priority, requests age in arrival order, so the best candidate overall is always among the $L$ queue heads, reducing selection to $O(L)$ for $L$ priority levels.

## In the wild

- **vLLM** (`--scheduling-policy priority`) orders its waiting queue in a heap by `(priority, arrival_time)` with lower values first. When KV memory runs out, it preempts the running request with the largest `(priority, arrival_time)`. Because vLLM has no built-in aging, deployments that need starvation protection adjust priorities at the gateway.
- **Triton**'s dynamic batcher supports `priority_levels` with per-level queue timeouts and size limits, also without aging. Requests that wait too long are rejected rather than promoted.
- **Operating systems** introduced aging in the Unix multilevel feedback queue, which periodically boosts the priority of waiting processes to prevent starvation.
- Related scheduling policies order requests by deadline (`slo-deadline-scheduler`) or by cumulative service per tenant (`vtc-fair-scheduler`).
