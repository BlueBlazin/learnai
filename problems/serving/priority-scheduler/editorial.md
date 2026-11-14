## The idea

Strict priority scheduling answers "who is most urgent?" and has a well-known failure:
if urgent work keeps arriving, less urgent work waits forever. Aging repairs it by
making urgency grow with waiting time. Here the growth is one level per `aging_ms`:

$$
\text{effective}(t) = \text{priority} - \left\lfloor \frac{t - \text{arrival}}{\text{aging\_ms}} \right\rfloor .
$$

A request with priority $p$ is at the level of a fresh priority-0 request after
$p \cdot \text{aging\_ms}$, and from then on it wins the tie against every request
that arrived after it, because ties go to the earlier arrival. So its extra wait,
compared with what a priority-0 request arriving at the same time would see, is
bounded by roughly $(p + 1) \cdot \text{aging\_ms}$: starvation becomes a tunable
delay.

The consequence for the implementation is that the rank of a waiting request is a
function of time. Two requests with different arrival times cross their level
boundaries at different instants, so their relative order can flip while nothing
arrives and nothing is admitted. A key stored at insertion is wrong the moment the
clock moves. The robust design evaluates the rule exactly when it is needed: when a
slot is free, scan the waiting list with the current time.

The tie-breaks are part of the rule, not decoration. Arrival time before queue
position makes the order independent of *when the scheduler happened to look* (a
request that arrived mid-step is not penalized), and never using the id keeps opaque
ids opaque.

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

The loop is unchanged from `continuous-batching-scheduler` except for the admission
line. That is the general shape of scheduling policies in an iteration-level engine:
the loop owns the mechanics, and a policy is a function that orders the waiting list.

## Common mistakes

- **No aging.** Returning the base priority passes every test of ordering at one
  instant and starves low priorities under a steady urgent stream: 4946 ms instead of
  1320 ms in the bound's workload, and unbounded as the stream gets longer.
- **A static sort key.** A heap keyed by `(priority, arrival)` at insertion cannot
  express aging at all. The usual patch, refreshing the keys whenever a request
  arrives, still decides on stale levels when a slot frees up with no arrival in
  between, which is precisely when the queue is draining and order matters.
- **Fractional aging.** `priority - waited / aging_ms` looks equivalent and is a
  different policy: a priority-1 request that has waited 90% of a level passes a
  fresh priority-0 request. (It is also a static order, `priority + arrival /
  aging_ms`, which is why it is tempting.) The statement asks for whole levels.
- **Off by one at the boundary.** At exactly `aging_ms` the level has been gained
  (`floor`, not `ceil - 1`). With arrival times and durations on a 0.25 ms grid these
  boundaries are hit exactly.
- **Waiting measured from collection.** Stamping a request with `loop.time()` when it
  leaves the queue forgets the time it sat there during a step. Use `arrival_s`.
- **Ties by id.** Sorting by `(effective, id)` works on every workload whose ids
  happen to increase with arrival time, which is every workload you would write by
  hand.
- **Larger is more urgent.** The convention here, as in vLLM and Unix `nice`, is that
  a smaller number is served first.

## Complexity

`pick_next` is a linear scan, $O(W)$ for $W$ waiting requests, once per admission.
That is negligible next to a forward pass of several milliseconds, and it is the
honest cost of a key that changes with time. If $W$ were in the hundreds of
thousands, the fix is one FIFO per base priority: within a level, requests age in
arrival order, so the best candidate overall is the best of the queue heads, $O(L)$
for $L$ levels.

## In the wild

- **vLLM** (`--scheduling-policy priority`): the waiting queue is a heap ordered by
  `(priority, arrival_time)`, lower first, and when KV memory runs out the running
  request with the *largest* `(priority, arrival_time)` is preempted. There is no
  aging, so a deployment that needs it adjusts priorities at the gateway.
- **Triton**'s dynamic batcher has `priority_levels` with a per-level queue policy
  (timeouts, maximum queue size) and no aging either; a request that times out in
  the queue is rejected or delayed, the blunt alternative to aging.
- **Operating systems** are where aging comes from: the Unix multilevel feedback queue
  periodically boosts the priority of processes that have waited, for the same reason.
- The next step in this direction is a scheduler that orders by *deadline* instead of
  by level (`slo-deadline-scheduler`), and one that orders by service received per
  tenant (`vtc-fair-scheduler`).
