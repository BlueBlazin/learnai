## The idea

A request now has a life cycle with three exits, and two of them are triggered from
outside the scheduling loop: `cancel()` is called by a client, and a deadline is
reached by the clock. Both can happen at any moment, including while the loop is
suspended in `await engine.step(...)`.

The design that makes this manageable has two parts.

**One exit door.** A single method ends a request, whatever the reason:

1. if it has already ended, do nothing and say so (this makes every race harmless:
   cancel after finish, a second cancel, a deadline after a cancel);
2. disarm its deadline timer;
3. take it out of the structure it is in: the waiting list, or the running list, and
   for an early end also out of the engine with `engine.release(id)`;
4. resolve its future, which wakes the client.

Because step 3 runs immediately, inside `cancel()` or the timer callback, the engine's
rule does the rest: a sequence released during a step gets nothing from that step. No
token can appear at or after the instant the request ended, and the slot count is
right when the loop plans the next step.

**The loop trusts current state.** Between starting a step and resuming after it, the
running list may have shrunk. So after the `await` the loop looks handles up in the
running list as it is *now*, and it relies on the engine's report (`result.tokens`,
`result.finished`), which never mentions a released sequence. It never assumes that
what it put into the step is still there.

The tie rule deserves a sentence. The engine completes a step after all ordinary
timers of the same instant, so a deadline at time $d$ beats a token due at $d$.
`loop.call_at` creates such a timer and its callback runs in that timer slot. A task
doing `await asyncio.sleep(...)` is also woken by a timer, but the task itself only
resumes afterwards, when the step's completion has already run: one line of difference
in the code, one token (or one wrongly "finished" request) of difference in the trace.

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

A finished sequence needs no `release`: the engine dropped it with its last token.
That is why `_end` releases only when the outcome is an exception.

## Common mistakes

- **Cancellation honored only while waiting.** Removing a request from the queue is
  the easy half. If a running request is merely marked as canceled for its client, it
  keeps its slot and generates to the end; the trace shows tokens after the cancel.
- **Leaking the slot.** A separate "slots in use" counter that is decremented where
  sequences *finish* misses the two other exits. Each canceled request then costs a
  slot forever, and after `max_running` of them the server accepts requests and never
  runs them. Derive the count from the running list instead.
- **Expiry checked only at admission.** `if now >= deadline: reject` at the head of
  the queue handles requests that waited too long, tells them late, and lets a running
  request generate past its deadline. A deadline is an event with its own timer.
- **Deferring the release to the next step boundary.** Setting a flag that the loop
  acts on later is the natural asyncio pattern and is one step late: the step in
  flight still produces the token, and its KV memory stays allocated meanwhile.
- **A stale timer.** If the deadline timer is not canceled when the request ends, and
  its callback does not check, it releases a sequence that finished long ago. The
  engine reports `unknown_sequence`; in a real server, where ids are reused, this
  kills somebody else's request.
- **Admitting a request that is already late.** `loop.call_at` with a time in the past
  fires soon, but not before the loop has admitted the request and started a step for
  it. Check in `submit`.
- **A deadline as a sleeping task.** See the tie rule above: the task resumes after the
  step's completion, so a token due at the deadline is produced.

## Complexity

`_end` is $O(W + R)$ for the `in` and `remove` on the waiting and running lists; the
loop is unchanged from continuous batching. If cancellations of long queues mattered,
a per-handle state field plus lazy deletion (skip ended handles when popping) makes
every operation $O(1)$. One timer per pending deadline is $O(\log n)$ in the event
loop's heap.

## In the wild

- **vLLM** watches the HTTP connection; on a disconnect the API server calls
  `abort(request_id)` on the engine, and the scheduler frees the request's KV blocks
  and drops it from `running` or `waiting` before the next step. Tokens of an aborted
  request that an in-flight step still computes are discarded, the same contract as
  `engine.release` here.
- **TGI** gives each request a channel to its client; when the client goes away the
  send fails and the router filters the request out of the running batch at the next
  iteration. **Triton** has per-request queue timeouts and a cancellation API that
  backends must poll.
- **gRPC** propagates a deadline with every call, and its status codes `CANCELLED` and
  `DEADLINE_EXCEEDED` are the two exceptions of this problem. OpenAI-style HTTP APIs
  have no deadline field; servers there rely on disconnect detection plus a
  server-side maximum request time.
- A production scheduler goes one step further and uses the deadline *before* it
  passes: a request that cannot finish in time even if admitted now is shed
  immediately (`slo-deadline-scheduler`), and on shutdown running requests are given a
  grace period before they are canceled (`graceful-drain`).
