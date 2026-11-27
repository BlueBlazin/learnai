## The idea

Graceful drain adds a three-state lifecycle on top of the serving loop:

| state | `ready` | `submit` | pending requests | ends when |
|---|---|---|---|---|
| serving | `True` | accepts | served | `drain()` is called |
| draining | `False` | raises `Unavailable` | served as before | nothing is pending and the engine is idle |
| closed | `False` | raises `Unavailable` | none | never |

Two events drive the state transitions. First, `drain()` switches from `serving` to `draining`. Because `drain()` is a synchronous method, the transition is atomic: no request can sneak in between setting `ready = False` and refusing new calls to `submit()`.

Second, the serving loop itself detects the transition to `closed`. Because `run()` is the only caller of `engine.step()`, the engine is guaranteed to be idle at the top of each loop iteration. The existing idle check (`not self._waiting and not self._running`) already identifies when nothing is pending, so completing the drain only requires canceling any active timer, setting `closed = True`, resolving the future, and returning.

The grace deadline is a third event that shortens the draining phase when work takes too long. Its callback terminates every pending request at once and releases all running sequences from the engine. Normal execution then finishes the job. If a step is in flight, it runs to completion (an active forward pass cannot be undone), and the loop resumes on the next iteration, sees both lists empty, and resolves the drain future. That is why the future can resolve after the deadline, and why resolving inside the deadline callback would wrongly declare the server idle while the engine is still busy.

Nothing about scheduling changes during a drain. Waiting requests were already accepted and their clients hold valid handles, so they continue to be admitted as slots open. Only the grace deadline breaks that promise, signaling the cut explicitly with `Terminated(tokens)`.

## Reference solution (Python)

```python
class Server:
    def __init__(self, engine, max_running):
        self.engine = engine
        self.max_running = max_running
        self.ready = True
        self.closed = False
        self._waiting = deque()
        self._running = []
        self._wake = asyncio.Event()
        self._drained = None  # the future drain() returned
        self._deadline = None  # the timer of the grace deadline
        self._terminated = 0

    def submit(self, request):
        if not self.ready:
            raise Unavailable("the server is draining")
        handle = Handle(request, asyncio.get_running_loop().create_future())
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def drain(self, grace_ms):
        loop = asyncio.get_running_loop()
        self.ready = False
        self._drained = loop.create_future()
        if grace_ms <= 0:
            self._terminate_pending()
        else:
            self._deadline = loop.call_at(loop.time() + grace_ms, self._terminate_pending)
        self._wake.set()
        return self._drained

    def _terminate_pending(self):
        for handle in self._running:
            self.engine.release(handle.request.id)  # also mid-step
        pending = [*self._running, *self._waiting]
        self._running.clear()
        self._waiting.clear()
        for handle in pending:
            handle._future.set_result(Terminated(handle._tokens))
        self._terminated += len(pending)
        self._wake.set()

    async def run(self):
        while True:
            if not self._waiting and not self._running:
                if self._drained is not None:
                    if self._deadline is not None:
                        self._deadline.cancel()
                    self.closed = True
                    self._drained.set_result(self._terminated)
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            ...  # admit, step, and account for the result as in the starter
```

`Handle.result()` awaits `self._future` and raises the value if it is an exception.

## Common mistakes

- **Dropping waiting requests when the drain starts.** Treating "in flight" as only "running on the GPU" fails queued requests that were already accepted and could have finished within the grace period.
- **Continuing to admit requests.** Flipping `ready = False` alone does not stop traffic because load balancers lag and clients reuse open connections. If `submit` keeps accepting, the server never drains under load and late requests get cut off instead of failing over immediately.
- **Resolving while streams are open.** Returning an already-resolved future from `drain()` tells the supervisor it can exit immediately, killing active engine work.
- **Resolving at the deadline while a step is in flight.** Releasing sequences mid-step does not end the GPU step early. Wait for `run()` to regain control after `engine.step()` returns before resolving the drain future.
- **Terminating handles without releasing sequences.** Failing client futures without calling `engine.release()` leaves sequences running in the engine until completion, wasting compute and delaying shutdown.
- **Sleeping for the full grace period.** Using `await asyncio.sleep(grace_ms)` before checking idle state wastes the entire grace period on idle servers or fast-finishing requests, slowing fleet rollouts.
- **Using a sleeping task or a zero-delay timer for the deadline.** Both execute after a step that completes at the same instant, letting a token due at the deadline slip through. Use `loop.call_at(deadline, callback)` for `grace_ms > 0`, and call `_terminate_pending()` synchronously inside `drain()` for `grace_ms == 0`.
- **Never returning from `run()`.** If the loop goes back to awaiting `self._wake` after resolving the drain future, the server task never exits.
- **Counting only released sequences.** The drain future must resolve to the total number of clients that received `Terminated`, including requests that were still waiting.

## Complexity

`drain` runs in $O(1)$ time, and `_terminate_pending` runs in $O(n)$ time for $n$ pending requests. The serving loop itself is unchanged except for one extra check in the idle branch.

## In the wild

- **Kubernetes** defines the outer shutdown protocol in its Pod Lifecycle ("Termination of Pods"). The kubelet runs the `preStop` hook, sends `SIGTERM` to PID 1, and sends `SIGKILL` once `terminationGracePeriodSeconds` (default 30 s) expires. In parallel, the control plane marks the pod `ready: false` in EndpointSlices. Because endpoint propagation races with `SIGTERM`, production servers often sleep briefly in `preStop` or allow a short lame-duck window before rejecting traffic. This problem models the `drain` phase after that window, where `grace_ms` must leave margin before `SIGKILL` arrives.
- **HTTP inference servers** reject post-drain requests with `503 Service Unavailable` (often with `Retry-After` or `Connection: close`) so clients (as in `retry-backoff-client`) retry on another replica. Once a streaming response has started, the HTTP status line has already been sent, so cutoffs are reported via an in-stream error event analogous to `Terminated.tokens`.
- **uvicorn** uses the same connection-level lifecycle on `SIGTERM`: it stops accepting new connections, waits for active requests, and cancels remaining tasks after `--timeout-graceful-shutdown`. An inference server adds engine-side sequence release so canceling an HTTP task also frees GPU work and KV cache blocks.
- In this track's simulation kit, `SimCluster.drain(replica)` provides the cluster-side counterpart used by `autoscaler-policy`, stopping new routing to a replica until its accepted work completes.
