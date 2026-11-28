## The idea

A request can end in three ways, and two of them happen outside the scheduling loop: a client calls `cancel()`, or the clock hits a deadline. Either event can fire while `run()` is suspended inside `await engine.step(...)`.

Two design principles keep the state consistent:

**One exit path.** Route every ending (normal completion, cancellation, and expiration) through a single `_end(handle, outcome)` method:

1. If the handle has already ended, return `False` immediately. This makes every race harmless (a cancel after completion, a duplicate cancel, or a deadline after a cancel).
2. Cancel the handle's deadline timer if one is active.
3. Remove the handle from `_waiting` or `_running`. If it was running and ended early with an exception, call `engine.release(id)`.
4. Resolve the handle's future so the waiting client wakes up.

Because step 3 runs immediately inside `cancel()` or the timer callback, the engine drops any mid-step sequence before that step finishes. No token is produced at or after the termination instant, and the slot is free when the next step is planned.

**Trust current state after `await`.** While `run()` is suspended in `engine.step(...)`, `_running` may shrink due to cancellations or deadlines. When the step resumes, the loop builds its `id -> handle` lookup from `_running` as it exists *now* and iterates over `result.tokens` and `result.finished`, which never include released sequences. It never assumes that every sequence passed into `engine.step(...)` is still running.

The same-instant tie rule depends on how `asyncio` schedules timers. The engine completes a step after all ordinary timers for that timestamp have run, so a deadline at time $d$ fires before a token due at $d$. `loop.call_at` schedules an ordinary timer whose callback runs in that slot. By contrast, a coroutine doing `await asyncio.sleep(...)` is woken by a timer, but the coroutine itself only resumes after the step completion callback has already run, producing an extra token or marking an expired request as finished.

## Reference solution (Python)

```python
class Handle:
    def __init__(self, scheduler, request, future):
        self.request = request
        self._scheduler = scheduler
        self._future = future      # resolves to a token count or an exception instance
        self._timer = None
        self._tokens = 0

    def cancel(self) -> bool:
        return self._scheduler._end(self, Canceled(f"request {self.request.id!r} was canceled"))

    async def result(self) -> int:
        outcome = await self._future
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Scheduler:
    def __init__(self, engine, max_running):
        self.engine = engine
        self.max_running = max_running
        self._waiting = deque()
        self._running = []
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        loop = asyncio.get_running_loop()
        handle = Handle(self, request, loop.create_future())
        deadline = request.deadline_s
        if deadline is not None and deadline <= loop.time():
            handle._future.set_result(DeadlineExceeded(f"request {request.id!r} arrived too late"))
            return handle
        if deadline is not None:
            expired = DeadlineExceeded(f"request {request.id!r} missed its deadline")
            handle._timer = loop.call_at(deadline, self._end, handle, expired)
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def _end(self, handle, outcome) -> bool:
        if handle._future.done():
            return False
        if handle._timer is not None:
            handle._timer.cancel()
        if handle in self._running:
            self._running.remove(handle)
            if isinstance(outcome, Exception):
                self.engine.release(handle.request.id)   # also mid-step
        else:
            self._waiting.remove(handle)
        handle._future.set_result(outcome)
        return True

    async def run(self) -> None:
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue
            decode = [h.request.id for h in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                handle = self._waiting.popleft()
                self.engine.admit(handle.request)
                prefill.append((handle.request.id, handle.request.prompt_tokens))
                self._running.append(handle)
            result = await self.engine.step(prefill=prefill, decode=decode)
            by_id = {h.request.id: h for h in self._running}
            for seq_id in result.tokens:
                by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                self._end(by_id[seq_id], by_id[seq_id]._tokens)
```

A normally finished sequence is already dropped by the engine when its last token is produced, so `_end` calls `engine.release` only when `outcome` is an exception.

## Common mistakes

- **Honoring cancellation only while waiting.** Removing a waiting request from the queue is not enough. If a running request is only marked canceled for its client without calling `engine.release`, it keeps its slot and continues decoding after cancellation.
- **Leaking slots.** Maintaining a separate `slots_in_use` counter and decrementing it only when sequences finish misses cancellations and deadlines. After `max_running` early exits, the server stops admitting new requests. Derive occupancy directly from `len(self._running)`.
- **Checking deadlines only at admission.** Rejecting expired requests at the head of the queue notifies waiting clients late and allows running requests to decode past their deadlines. Deadlines need their own timers.
- **Deferring `engine.release` to the next step boundary.** Setting a flag for the loop to handle after `await engine.step(...)` is one step too late. The in-flight step still emits a token and holds KV memory until it finishes.
- **Leaving stale timers armed.** If a deadline timer is not canceled when a request finishes or is canceled, and the callback does not guard against already-ended handles, it later calls `engine.release` on a finished sequence. Here that raises `unknown_sequence`; in a real server that reuses request ids, it aborts someone else's request.
- **Admitting an already-expired request.** Scheduling `loop.call_at` with a past timestamp fires soon, not immediately, so the loop can admit the request first. Reject `deadline_s <= loop.time()` inside `submit`.
- **Implementing deadlines with a sleeping task.** As noted above, `await asyncio.sleep(...)` resumes after the step completion callback, letting a token due at the exact deadline slip through.

## Complexity

`_end` takes $O(W + R)$ time for membership checks and removal on `_waiting` ($W$ items) and `_running` ($R$ items). The scheduling loop itself has the same cost as basic continuous batching. For very large queues, adding a per-handle state field and using lazy deletion (skipping ended handles when popping from `_waiting`) makes every operation $O(1)$. Each active deadline timer costs $O(\log n)$ in the event loop's timer heap, where $n$ is the number of active timers.

## In the wild

- **vLLM** monitors each client's HTTP connection. On disconnect, the API server calls `abort(request_id)` on the engine, and the scheduler frees the request's KV blocks and removes it from `running` or `waiting` before the next step. Any tokens computed by an in-flight step for an aborted request are discarded, matching `engine.release` here.
- **TGI** attaches a channel to each client; when a send fails, the router removes the request from the running batch on the next iteration. **Triton** provides per-request queue timeouts and a cancellation API that backends poll.
- **gRPC** propagates a deadline on every RPC, and its `CANCELLED` and `DEADLINE_EXCEEDED` status codes correspond directly to the two exceptions here. OpenAI-style HTTP APIs lack a standard deadline field, so servers combine disconnect detection with a server-side maximum request duration.
- Production schedulers build on this by checking feasibility before a deadline passes, shedding requests that cannot finish in time (`slo-deadline-scheduler`), and giving running requests a grace period during shutdown (`graceful-drain`).
