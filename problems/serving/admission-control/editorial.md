## The idea

Admission control is a promise made at the door. A server with one slot that
processes $r$ prompt tokens per second serves an admitted request after everything
ahead of it: the request that is running, and the queue. If the queue never holds
more than $T_{\max}$ prompt tokens, a newly admitted request finishes within

$$
\frac{T_{\max} + p_{\text{longest}}}{r}
$$

seconds, whatever the arrival rate. The bound on tokens is what makes the promise;
the bound on requests covers the per-request costs that tokens do not measure
(memory for request state, scheduling overhead). Without either, the wait is
proportional to how long the overload has lasted.

Three details carry the correctness:

1. **Decide, then enqueue.** The test is on the state *with* the new request,
   $Q + 1 \le Q_{\max}$ and $T + p \le T_{\max}$, and a rejection must leave nothing
   behind.
2. **Queued means waiting.** A request stops counting at `start`. The bounds exist to
   limit waiting; work in progress is limited elsewhere, by the engine's batch size
   and memory.
3. **An honest `Retry-After`.** A client that retries immediately is rejected again
   and adds load; one that is told to wait until the whole queue is empty stays away
   too long. The right estimate is the time until *enough* of the queue has drained
   for this request to fit, which depends on the request: a small prompt needs one
   departure, a large one may need several, and one that exceeds the token bound by
   itself needs to be told never.

## Reference solution (Python)

```python
class Admission:
    def __init__(self, max_queued_requests, max_queued_tokens, drain_tokens_per_s):
        self.max_queued_requests = max_queued_requests
        self.max_queued_tokens = max_queued_tokens
        self.drain_tokens_per_s = drain_tokens_per_s
        self._queue = {}            # id -> prompt_tokens, in arrival order
        self._queued_tokens = 0
        self._running = set()

    def admit(self, req):
        if self._fits(req.prompt_tokens, len(self._queue), self._queued_tokens):
            self._queue[req.id] = req.prompt_tokens
            self._queued_tokens += req.prompt_tokens
            return Admitted()
        return Rejected(self._retry_after_s(req.prompt_tokens))

    def start(self, req):
        self._queued_tokens -= self._queue.pop(req.id)
        self._running.add(req.id)

    def finish(self, req):
        self._running.remove(req.id)

    def _fits(self, prompt_tokens, queued_requests, queued_tokens):
        return (queued_requests + 1 <= self.max_queued_requests
                and queued_tokens + prompt_tokens <= self.max_queued_tokens)

    def _retry_after_s(self, prompt_tokens):
        if prompt_tokens > self.max_queued_tokens:
            return None
        requests, tokens, drained = len(self._queue), self._queued_tokens, 0
        for waiting in self._queue.values():
            requests -= 1
            tokens -= waiting
            drained += waiting
            if self._fits(prompt_tokens, requests, tokens):
                break
        return drained / self.drain_tokens_per_s
```

The three properties return `len(self._queue)`, `self._queued_tokens`, and
`len(self._running)`.

## Common mistakes

- **No bound at all.** The default of many servers. Every request is accepted and
  the queue absorbs the overload; in the 3x overload test the last requests would
  wait about two minutes for a server whose requests take half a second.
- **Checking after enqueueing.** Appending first and testing `len(queue) > max` is
  equivalent only if the rejected request is removed again. When it is not, rejected
  requests occupy the queue forever and the server soon rejects everything.
- **Counting running requests as queued.** Releasing a request's tokens at `finish`
  instead of `start` makes the token bound cover work in progress: with a long prompt
  running, the server rejects requests although nothing is waiting.
- **`<` for `<=`.** A request that fills the queue exactly is within the bounds.
  Rejecting it wastes one slot permanently, and `max_queued_requests=1` rejects
  everything.
- **Bounding requests only.** Eight queued requests of 100,000 tokens are a very
  different wait from eight chat turns.
- **`Retry-After` for the whole queue**, or for one departure regardless of the
  request's size. The first keeps clients away from a server that has room; the
  second sends large requests back too early, to be rejected again.
- **A finite `Retry-After` for a request that can never fit.** The client will retry
  forever. Say `None` here, and a 4xx rather than a 503 at the HTTP layer.
- **Assuming FIFO starts.** `start` names the request. A scheduler that picks by
  priority or by length removes requests from the middle of the queue.

## Complexity

`admit` is $O(1)$ when it accepts and $O(Q)$ when it rejects (the scan for the retry
estimate, at most `max_queued_requests` steps). `start` and `finish` are $O(1)$.
Memory is $O(Q + \text{running})$.

## In the wild

- **vLLM** has both knobs under the names this problem uses, with one difference in
  what they count. `--max-num-queued-reqs` is documented as the "maximum number of
  requests that can be in-flight (waiting or running) at the same time", and
  `--max-num-queued-tokens` as the "maximum total prompt tokens of requests currently
  in the prefill phase"; beyond either, new requests are rejected with HTTP 503 "so
  the client can retry on another instance", and rejections are counted in the
  `vllm:admission_rejections` metric
  ([engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html),
  [metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)). This problem bounds
  waiting requests only, which keeps the latency argument above exact; with vLLM's
  definition, set the request limit to the running capacity plus the queue you want.
- **TGI** rejects with HTTP 429 when its queue exceeds `--max-concurrent-requests`,
  and validates `--max-input-tokens` and `--max-total-tokens` per request, the
  analogue of `Rejected(None)`. **Triton** bounds each model's queue with
  `max_queue_size` and can time requests out of the queue.
- **Where the drain rate comes from.** Here it is a constructor argument. A real
  server measures it: an exponentially weighted average of prefill tokens per second
  from the engine's own counters. The estimate is only as good as that number, which
  is why `Retry-After` is advice, and why clients add jitter to it.
- **Rejection is the cheap half of backpressure.** The other half is the client's
  behavior: exponential backoff with jitter, honoring `Retry-After`, and a retry
  budget. A fleet behind a load balancer also exports the queue depth, so the
  balancer can stop sending to a replica before it has to reject.
