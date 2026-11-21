## The idea

A gateway is an event handler with a queue. Three things happen to it: a request
arrives, a request completes, the autoscaling policy is evaluated. Each of them
ends in the same routine, `dispatch`, which moves requests from the gateway queue
to replicas for as long as some ready replica has a free slot. Inside `dispatch`
there are two choices, **which request** and **which replica**, and the stages are
the policies behind those choices plus the two decisions at the edges: whether a
request gets into the queue at all, and how many replicas there are.

![A request passes admission (its tenant's token bucket, then the bounds of the queue) or is rejected at once; admitted requests wait in the gateway queue until a ready replica has a free slot, and the dispatcher then chooses which request leaves the queue and which replica gets it. Completions come back to the gateway, which also evaluates the autoscaling policy.](assets/architecture.svg)

Why hold a queue in the gateway at all, when every replica has one? Because a
request that has been sent can no longer be reordered or rerouted. Keeping each
replica at `replica_slots` requests and the rest at the gateway is what gives the
other stages something to decide: fairness needs a queue to choose from, and a
request that waits at the gateway can still go to the replica that becomes free
first, or to one that did not exist when it arrived.

**Stage 1** is `admission-control` with a `token-bucket-limiter` per tenant in
front. The order of operations is the content: compute the balance, check it,
check the queue bounds, and only then charge and enqueue, so that a rejection
leaves no trace. The bounds apply to the gateway's queue; requests on replicas are
bounded by `replica_slots` already.

**Stage 2** is one `min` with the key `(outstanding_tokens, id)` over the replicas
that have a free slot. Tokens, not requests, because that is the unit of work.

**Stage 3** is the router of `prefix-aware-router` with load measured in tokens.
The index is the gateway's belief about the replicas' caches (it cannot see them),
bounded like the caches are, and the eligibility test is what keeps a popular
prefix from burying one replica: affinity is followed only among replicas within
`imbalance_tokens` of the idlest. When it is broken, the prompt is recorded for
the replica it went to, and the prefix now lives in two places.

**Stage 4** is the Virtual Token Counter of `vtc-fair-scheduler` moved from the
engine's waiting list to the gateway's queue. The pick is a `min` over the queue
keyed by the tenant's counter (Python's `min` returns the first of equal keys,
which is the tie rule). Prompt tokens are charged at dispatch; output tokens when
the answer is complete, because that is when a gateway learns them. The lift keeps
a returning tenant from spending credit it "saved" while idle.

**Stage 5** is `autoscaler-policy` with one change in its input: the load is the
gateway queue plus what the cluster reports. With a gateway queue, the replicas
are pinned at `replica_slots` requests each, so the cluster's own numbers say
"full" whether ten requests wait or a thousand; the backlog is the signal.
Starting replicas count as capacity, the window delays scale-down, and
`scale_to` drains instead of stopping. The routing rule already skips replicas
that are not ready, which is all graceful scale-down asks of the gateway.

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

`_choose`, `_next_request`, `_lift`, `_finished`, and `evaluate` are the
prerequisites' few lines each; the third hint shows the last two.

The completion is handled in a done callback because a callback runs to its end
before anything else touches the gateway: charge, resolve, dispatch are one
atomic step. A task that awaits the future wakes up later, possibly after another
completion has been delivered, and then two slots are filled with counters that
are one charge behind.

## Common mistakes

- **An unbounded queue.** Every request is admitted and all of them are late:
  on the stage 1 workload 19 of 600 meet the SLO instead of 159. A variant bounds
  requests and forgets tokens.
- **Counting running requests as queued.** The gateway then rejects while its
  queue is empty.
- **A bucket without its cap, or charged too early.** Without `min(burst, ...)`
  an idle tenant saves up an unlimited burst; charging before the queue check
  makes a `queue_full` rejection cost tokens.
- **Request-count balancing.** One long document counts as much as one short
  question, so short requests queue behind long ones.
- **Ignoring `replica_slots`.** Everything is sent at once; the gateway queue is
  always empty and stages 4 and 5 have nothing to work with.
- **Sticky routing.** Following the best match whatever the load sends every
  request with the popular prefix to one replica.
- **An index that is not updated on fallback, or never forgets.** The first loses
  all cache hits (nothing is ever remembered); the second follows matches the
  replica evicted long ago.
- **Tenant starvation.** First come, first served: on the stage 4 workload none
  of the interactive tenant's 60 requests is on time.
- **No lift, request counting, output not charged.** Each breaks fairness for a
  different tenant shape: the returning one, the one with long prompts, the one
  with long answers.
- **Scale-down that kills in-flight requests.** `cluster.stop` instead of
  draining: the requests on the removed replicas fail.
- **Pending capacity or the gateway queue ignored, no window.** The autoscaler
  orders the same replicas again at every evaluation, never sees the backlog, or
  removes replicas in every lull.

## Complexity

Admission is $O(1)$. A dispatch is $O(R)$ for $R$ replicas plus the request
choice, $O(Q)$ for a queue of $Q$ with the linear `min` (a heap per tenant makes it
$O(\log)$), plus $O(R \cdot k \cdot L)$ token comparisons for prefix matching with
an index of $k$ prompts of length $L$. An evaluation is $O(R)$.

## In the wild

- The Virtual Token Counter is from "Fairness in Serving Large Language Models"
  (OSDI 2024, arXiv:2401.00588); the counter lift is its rule for clients that
  return from idleness.
- The endpoint picker of the Kubernetes Gateway API Inference Extension, which
  llm-d's scheduler builds on, scores model-server pods with plugins that include
  a queue scorer and a prefix-cache scorer, the latter from the gateway's own
  record of which prefixes it sent where (hashed in blocks, with an LRU capacity
  per server): stage 3's index.
- The scaling rule is the one of the Kubernetes Horizontal Pod Autoscaler,
  `desired = ceil(current * metric / target)` with a tolerance and a scale-down
  stabilization window, fed with requests waiting plus running.
