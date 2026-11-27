## The idea

Admission control is a latency promise enforced at the front door. Suppose a single-slot server processes $r$ prompt tokens per second. An admitted request waits behind whatever is currently running and whatever is already in the queue. If no prompt exceeds $p_{\text{longest}}$ tokens and the queue never holds more than $T_{\max}$ prompt tokens, every admitted request finishes within

$$
\frac{T_{\max} + p_{\text{longest}}}{r}
$$

seconds, regardless of how high the arrival rate spikes. The token bound guarantees that latency ceiling, while the request-count bound caps per-request overhead such as request metadata and scheduling work. Without both bounds, wait times grow in proportion to how long the overload lasts.

Three implementation rules make the controller correct:

1. **Check before enqueueing.** Evaluate both bounds with the new request included ($Q + 1 \le \text{max\_queued\_requests}$ and $T + p \le \text{max\_queued\_tokens}$). If either fails, reject without mutating state.
2. **Count only waiting requests.** A request stops counting against the queue bounds at `start`. These bounds limit waiting work, whereas in-flight work is governed separately by the engine's batch size and memory limits.
3. **Return an accurate `retry_after_s`.** Telling a client to retry immediately triggers another rejection and adds load. Telling it to wait until the entire queue empties keeps it away longer than necessary. The right estimate is the time needed for *enough* of the queue to drain so this specific request fits. A small prompt may need one departure, a large prompt may need several, and a prompt larger than `max_queued_tokens` can never fit and must receive `None`.

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

The three properties return `len(self._queue)`, `self._queued_tokens`, and `len(self._running)`.

## Common mistakes

- **No bound at all.** Many servers default to unbounded queues. In the 3x overload simulation, an unbounded queue forces the last requests to wait roughly two minutes on a server whose individual requests take half a second.
- **Checking after enqueueing.** Appending first and checking `len(queue) > max` only works if you roll back on rejection. Forgetting the rollback leaves rejected requests stuck in the queue until the server rejects all traffic.
- **Counting running requests as queued.** Releasing a request's tokens in `finish` instead of `start` counts active execution against the queue bound. When a long prompt is running, the server rejects new traffic even though nothing is waiting.
- **Using `<` instead of `<=`.** A request that brings the queue exactly to its limit must be admitted. Using strict `<` permanently wastes one slot and causes `max_queued_requests=1` to reject every request.
- **Bounding request count only.** Eight queued 100,000-token prompts create a vastly longer wait than eight short chat turns.
- **Bad `Retry-After` estimates.** Returning the drain time of the entire queue keeps clients away after room has opened up. Returning only one departure's drain time regardless of `prompt_tokens` sends large requests back too early to be rejected again.
- **Returning a finite `Retry-After` for an oversized prompt.** If `prompt_tokens > max_queued_tokens`, returning a number makes the client retry forever. Return `None` here, which maps to an HTTP 4xx error rather than a retryable 503.
- **Assuming FIFO starts.** `start(req)` names a specific request. A priority or shortest-job-first scheduler removes requests from the middle of the queue, so a plain FIFO pop fails.

## Complexity

`admit` takes $O(1)$ time on acceptance and $O(Q)$ time on rejection to scan the queue (at most `max_queued_requests` steps). `start` and `finish` take $O(1)$ time. Memory is $O(Q + R)$ for $Q$ queued and $R$ running requests.

## In the wild

- **vLLM** exposes both limits under similar names, with one difference in accounting. `--max-num-queued-reqs` caps the maximum number of in-flight requests (waiting or running), while `--max-num-queued-tokens` caps the total prompt tokens currently in the prefill phase. Requests beyond either limit receive HTTP 503 so clients can retry on another instance, and rejections increment `vllm:admission_rejections` ([engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html), [metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)). This problem bounds waiting requests only to keep the latency formula exact. Under vLLM's definition, set the request limit to your running capacity plus your desired queue depth.
- **TGI** returns HTTP 429 when its queue exceeds `--max-concurrent-requests` and validates `--max-input-tokens` and `--max-total-tokens` per request (the analogue of `Rejected(None)`). **Triton** bounds each model's queue with `max_queue_size` and can expire requests that wait too long.
- **Estimating the drain rate.** Here `drain_tokens_per_s` is passed to the constructor. Real servers measure it dynamically using an exponentially weighted moving average of prefill tokens per second from engine counters. Because that rate is an estimate, `Retry-After` is advisory and clients add jitter before retrying.
- **Client-side backpressure.** Server-side rejection is only half of overload protection. Clients pair it with exponential backoff, jitter, `Retry-After` handling, and retry budgets. Behind a load balancer, replicas also export queue depth so traffic shifts away before rejections start.
