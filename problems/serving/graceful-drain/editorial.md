## The idea

A drain is a small state machine laid over the serving loop:

| state | `ready` | `submit` | pending requests | ends when |
|---|---|---|---|---|
| serving | `True` | accepts | served | `drain()` is called |
| draining | `False` | raises `Unavailable` | served as before | nothing is pending and the engine is idle |
| closed | `False` | raises `Unavailable` | none | never |

Two events move it. `drain()` is the first, and because it is a plain method the
transition is atomic: no request can slip in between "readiness is false" and "new
work is refused". The second is "idle", and the design question is who detects it.
The answer is the loop itself. It is the only caller of `engine.step`, so at the top
of an iteration the engine is not busy, and its own two lists say whether anything
is pending. The existing idle branch (`not waiting and not running`) is exactly the
condition, so the drain adds three lines there: resolve, mark closed, return.

The grace deadline is a third event that only shortens the draining state. It ends
every pending request at once, the running ones with `engine.release`, and then the
normal path takes over: if a step is in flight it still runs to its end (a forward
pass cannot be taken back), the loop resumes, finds both lists empty, and resolves.
That is why the future can resolve *after* the deadline, and why resolving inside
the deadline callback would be wrong: the engine would still be computing.

Nothing about serving changes during the drain. Waiting requests were accepted; the
client holds a handle and was promised an answer, so they keep being admitted. Only
the deadline may break that promise, and it does so explicitly, with `Terminated`
and the count of tokens the client already has.

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

`Handle.result()` awaits the future and raises the value if it is an exception.

## Common mistakes

- **Dropping the waiting requests when the drain starts.** "In flight" is read as
  "running on the GPU". A request in the waiting list was accepted too; its client
  cannot tell the difference and sees a failure although the grace period had room.
- **Still admitting.** Flipping the readiness flag does not stop traffic: load
  balancers react late, and clients with open connections do not look at probes. If
  `submit` keeps accepting, the server never becomes idle under load, and every late
  request is cut off at the deadline instead of being sent elsewhere at once.
- **Resolving while streams are open.** A `drain` that sets a flag and returns a
  finished future tells the supervisor it may exit; the process dies with requests
  in the engine.
- **Resolving at the deadline while the engine is busy.** After the releases, the
  step in flight still occupies the GPU. Report "idle" from the loop, when the step
  has ended.
- **Telling the clients without releasing the sequences.** The handles raise
  `Terminated`, but the sequences stay in the engine and generate to the end: tokens
  after the cut-off, and a drain that lasts as long as the longest request.
- **Sleeping out the grace period.** `await asyncio.sleep(grace)` then cancel
  everything is correct only at the deadline. On an idle server, or when the last
  request finishes after two seconds of a thirty-second grace period, it wastes the
  rest, and rollouts take the sum of all grace periods.
- **The deadline as a sleeping task, or as a timer for "now".** Both run after the
  engine has completed a step that ends at the same instant, so a token due at the
  deadline is produced, and a request finishing there counts as finished. A plain
  `call_at` timer fires before the step completes; for `grace_ms = 0` nothing is
  early enough except doing the work inside `drain`.
- **A loop that never returns.** If `run()` goes back to waiting after resolving the
  future, the process has no reason to exit and is killed at the end of the
  supervisor's own grace period.
- **Counting only what was released.** The value of the future is the number of
  clients that received `Terminated`, waiting ones included.

## Complexity

`drain` is $O(1)$ and `_terminate_pending` is linear in the number of pending
requests. The loop is unchanged; the drain adds one comparison per idle check.

## In the wild

- **Kubernetes** (Pod Lifecycle, "Termination of Pods") defines the outer protocol.
  The kubelet runs the container's `preStop` hook, then sends `TERM` to process 1,
  and sends `KILL` when the grace period (`terminationGracePeriodSeconds`, 30 seconds
  by default) expires; the `preStop` hook runs inside that same period. "At the same
  time as the kubelet is starting graceful shutdown", the control plane takes the pod
  out of the Service's EndpointSlices: terminating endpoints are reported with
  `ready: false`. These two things race, so a server can receive requests for a
  short while after `SIGTERM`. That is why real servers either sleep a few seconds
  in `preStop` before anything else, or keep accepting for a short "lame duck"
  interval after failing readiness; this problem's `drain` is what runs after that
  interval, and `grace_ms` must be chosen below what is left of the pod's grace
  period, or `SIGKILL` does the termination without the `Terminated` events.
- **HTTP.** The refusal is a `503 Service Unavailable`, usually with a `Retry-After`
  header or `Connection: close`, so that the client (see `retry-backoff-client`)
  retries and the load balancer sends the retry to another replica. For a stream
  that has started, the status line is long gone; the cut-off has to be an error
  event inside the stream, which is what `Terminated.tokens` stands for: the client
  learns that the output is incomplete and how much of it is valid.
- **uvicorn** implements the same shape for connections: on `SIGTERM` it stops
  accepting, waits for running requests, and `--timeout-graceful-shutdown` is the
  deadline after which the remaining tasks are canceled. An inference server needs
  the engine-side half on top (releasing sequences and their KV blocks), because a
  canceled HTTP task does not by itself stop a forward pass.
- The kit's `SimCluster.drain(replica)` is the cluster-side counterpart used by
  `autoscaler-policy`: a draining replica gets no new work and stops when what it
  holds has finished.
