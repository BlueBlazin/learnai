## The idea

An inference gateway reacts to three events: request arrival (`submit`), request completion (done callback), and autoscaling evaluation (`evaluate`). Each event ends by calling a single `_dispatch` routine that moves queued requests onto ready replicas as long as a free slot exists. Inside `_dispatch`, there are two policy decisions (which queued request to pick next and which replica should run it), surrounded by admission control at entry and replica scaling at the fleet boundary.

![A request passes admission or is rejected at once. The dispatcher pairs a queued request with a ready replica that has a free slot.](assets/architecture.svg)

Holding a queue in the gateway even though replicas have internal queues is essential. Once a request is handed to a replica, it can no longer be reordered or rerouted. Capping each replica at `replica_slots` keeps excess backlog in the gateway where fair scheduling can reorder tenants and late-arriving replicas can absorb queued work.

**Stage 1** combines `admission-control` with a per-tenant `token-bucket-limiter`. Check before mutating: compute the refilled balance, check the rate limit, check the gateway queue bounds, and only then deduct tokens and enqueue. That guarantees rejected requests leave no side effects. Queue bounds apply only to the gateway queue because in-flight work on replicas is already capped by `replica_slots`.

**Stage 2** selects the least-loaded ready replica with a free slot using `min` keyed by `(replica.outstanding_tokens, replica.id)`. Measuring load in tokens rather than request count reflects actual prefill and decode work.

**Stage 3** applies `prefix-aware-router` with `outstanding_tokens` as the load metric. Because the gateway cannot inspect replica KV caches, it maintains a bounded per-replica history of recently routed prompts. The `imbalance_tokens` eligibility check prevents a popular prefix from saturating one replica. When affinity breaks and a request falls back to the least-loaded replica, recording the prompt there lets that replica serve future hits from its own cache.

**Stage 4** moves the Virtual Token Counter from `vtc-fair-scheduler` to the gateway queue. Choosing the next request is a `min` over the queue keyed by `self.counters[req.tenant]` (Python's `min` returns the first minimum element, matching the FIFO tie-breaker). Prompt tokens are charged immediately at dispatch, while output tokens are charged in the completion callback once `output_tokens` is known. The counter lift prevents an idle tenant from accumulating unfair credit while away.

**Stage 5** runs `autoscaler-policy` on `len(self._queue) + m.queue_depth + m.running`. Including the gateway queue is critical because replicas stay pinned at `replica_slots` under overload, hiding backlog depth from `cluster.metrics()`. Counting `starting` replicas prevents double-ordering during the 4-second boot window, the stabilization window damps scale-down flapping, and `cluster.scale_to` drains replicas gracefully while stage 2's `"ready"` filter keeps new work off draining replicas.

## Reference solution (Python)

```python
def submit(self, request):
    cfg, now = self.cfg, self.cluster.sim.now()
    future = self.cluster.sim.loop.create_future()
    cost = request.prompt_tokens
    balance = None
    if cfg.rate_tokens_per_s is not None:
        balance = self._balance(request.tenant, now)  # min(burst, old + rate * dt)
        if balance < cost:
            retry = (cost - balance) / cfg.rate_tokens_per_s if cost <= cfg.burst_tokens else None
            future.set_result(Rejected("rate_limited", retry))
            return future
    if (
        len(self._queue) + 1 > cfg.max_queued_requests
        or self._queued_tokens + cost > cfg.max_queued_tokens
    ):
        future.set_result(Rejected("queue_full"))
        return future
    if balance is not None:
        self._buckets[request.tenant] = (balance - cost, now)  # charge only now
    if cfg.fair:
        self._lift(request.tenant)
    self._queue.append(request)
    self._queued_tokens += cost
    self._clients[request.id] = future
    self._dispatch()
    return future


def _dispatch(self):
    while self._queue:
        free = [
            r
            for r in self.cluster.replicas
            if r.state == "ready" and r.queue_len < self.cfg.replica_slots
        ]
        if not free:
            return
        request = self._next_request()  # oldest, or VTC
        self._queue.remove(request)
        self._queued_tokens -= request.prompt_tokens
        if self.cfg.fair:
            self.counters[request.tenant] += self.cfg.w_in * request.prompt_tokens
            self._last = request.tenant
        replica = self._choose(request, free)  # least loaded, or prefix-aware
        outcome = replica.submit(request)
        outcome.add_done_callback(lambda done, request=request: self._finished(request, done))
```

`_choose`, `_next_request`, `_lift`, `_finished`, and `evaluate` each take only a few lines; the third hint shows `_finished` and `evaluate`.

Handling completions in a done callback rather than an `await` task keeps charging, resolving, and dispatching atomic. An `await` task can resume after another completion fires, causing dispatches to read stale VTC counters.

## Common mistakes

- **Unbounded queue.** Admitting every request causes all replies to miss the SLO (19 good requests instead of 159 on the stage 1 workload). Bounding request count while forgetting `max_queued_tokens` causes the same failure on long prompts.
- **Counting running requests as queued.** Including replica-held requests in `max_queued_requests` or `max_queued_tokens` rejects traffic while the gateway queue is empty.
- **Uncapped or prematurely charged buckets.** Omitting `min(burst, ...)` lets idle tenants hoard unlimited bursts. Charging before checking queue bounds drains tokens on `queue_full` rejections.
- **Request-count load balancing.** Treating a long document and a short query as equal load traps short requests behind long prefills.
- **Ignoring `replica_slots`.** Forwarding everything immediately leaves the gateway queue empty, breaking fair scheduling and hiding backlog.
- **Unbounded sticky routing.** Ignoring `imbalance_tokens` piles every request sharing a hot prefix onto one replica.
- **Stale or unupdated prefix index.** Skipping index updates on least-loaded fallback loses cache hits on the new replica, while an unbounded index follows prefixes evicted from the replica's LRU cache long ago.
- **FIFO starvation.** Skipping VTC lets a batch tenant starve an interactive tenant (0 of 60 good chat requests on the stage 4 workload).
- **Incomplete VTC accounting.** Omitting the counter lift, counting requests instead of weighted tokens, or forgetting to charge `output_tokens` breaks fairness for returning tenants, long-prompt tenants, or long-output tenants.
- **Killing in-flight requests on scale-down.** Calling `cluster.stop` instead of `cluster.scale_to` fails active requests.
- **Autoscaler blind spots.** Ignoring `starting` replicas causes repeated over-scaling during boot, ignoring the gateway queue hides overload, and omitting the stabilization window causes rapid scale-down flapping during lulls.

## Complexity

Admission runs in $O(1)$. Each dispatch takes $O(R)$ to scan $R$ replicas, $O(Q)$ for linear VTC selection over a queue of size $Q$ (reducible to $O(\log U)$ over $U$ queued tenants with per-tenant queues and a heap), and $O(R \cdot k \cdot L)$ for prefix matching against an index of $k$ prompts of length $L$. Each `evaluate` call runs in $O(R)$.

## In the wild

- The Virtual Token Counter and its idle-client counter lift come from "Fairness in Serving Large Language Models" (OSDI 2024, arXiv:2401.00588).
- The Kubernetes Gateway API Inference Extension endpoint picker (used by llm-d's scheduler) scores model-server pods with queue-length and prefix-cache scorers, tracking routed prefixes in a bounded block-hash LRU index per pod just like stage 3.
- The autoscaling rule mirrors the Kubernetes Horizontal Pod Autoscaler (`desired = ceil(current * metric / target)` with tolerance and a scale-down stabilization window) driven by queued plus running requests.
