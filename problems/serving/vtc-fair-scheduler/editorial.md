## The idea

Fair queueing gives every backlogged tenant the same amount of *service*, regardless of how each tenant splits that service across requests. In VTC, service is measured in weighted tokens using $w_\text{in}$ per prompt token and $w_\text{out}$ per generated token (1 and 2 by default, matching typical API pricing). The scheduler tracks one number per tenant: its virtual token counter $c_i$, representing the cumulative weighted tokens charged to tenant $i$.

**Serve the smallest counter.** Whenever a slot opens, the queued tenant with the smallest counter has received the least service and goes next. Serving it raises its counter, so queued tenants continually overtake one another and stay balanced.

**Charge at token granularity.** Prompt tokens are charged immediately at admission so that any remaining picks in the same step see the updated counter. Otherwise, one tenant could claim every free slot in a single step. Output tokens are charged one by one as each step finishes. This removes any need to predict output lengths, because a tenant generating a long answer has its counter incremented on every step.

**Lift counters on return.** A counter that only increases turns idleness into saved credit. A tenant that sends nothing for an hour would have a counter far below active tenants and would monopolize the server upon returning until it caught up. Fairness applies only among tenants competing *now*, so unused past capacity cannot be banked. When an unqueued tenant submits a request, its counter is raised to the minimum counter among currently queued tenants, placing it at the current frontier instead of ahead of everyone. If no tenant is queued, the reference point is the counter of the tenant admitted most recently. Taking the `max` with the tenant's own counter ensures the lift only raises counters and never lowers them, so a heavily served tenant cannot erase its debt by briefly draining its queue.

Let $L_\text{input}$ be the longest prompt and $M$ be the maximum tokens in a running batch. Under these rules, the counters of queued tenants never differ by more than $U = \max(w_\text{in} L_\text{input},\ w_\text{out} M)$ (Lemma 4.3). Because a backlogged tenant's service over an interval equals the growth of its counter, any two backlogged tenants differ in service by at most $2U$ over any interval (Theorem 4.4). Crucially, this bound is independent of the interval length.

## Reference solution (Python)

```python
class VTCScheduler:
    def __init__(self, engine, max_running, w_in=1.0, w_out=2.0):
        self.engine, self.max_running = engine, max_running
        self.w_in, self.w_out = w_in, w_out
        self.counters = {}
        self._waiting = []       # Q, in arrival order
        self._running = []       # ids, in admission order
        self._tenant = {}        # live id -> tenant
        self._last = None        # tenant of the last request that left Q
        self._wake = asyncio.Event()

    def submit(self, request):
        tenant = request.tenant
        counter = self.counters.setdefault(tenant, 0.0)
        queued = {r.tenant for r in self._waiting}
        if tenant not in queued:
            if queued:
                counter = max(counter, min(self.counters[t] for t in queued))
            elif self._last is not None:
                counter = max(counter, self.counters[self._last])
            self.counters[tenant] = counter
        self._waiting.append(request)
        self._wake.set()

    def _pick(self):
        request = min(self._waiting, key=lambda r: self.counters[r.tenant])
        self._waiting.remove(request)
        return request

    async def run(self):
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue
            decode = list(self._running)
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                request = self._pick()
                self.counters[request.tenant] += self.w_in * request.prompt_tokens
                self._last = request.tenant
                self._tenant[request.id] = request.tenant
                self.engine.admit(request)
                prefill.append((request.id, request.prompt_tokens))
                self._running.append(request.id)
            result = await self.engine.step(prefill=prefill, decode=decode)
            for seq_id in result.tokens:
                self.counters[self._tenant[seq_id]] += self.w_out
            for seq_id in result.finished:
                self._running.remove(seq_id)
                del self._tenant[seq_id]
```

Calling `min` over `self._waiting` with `key=lambda r: self.counters[r.tenant]` returns the first request in arrival order among tenants tied for the minimum counter. That single expression satisfies both the tie-breaking rule and per-tenant FIFO ordering.

## Common mistakes

- **Omitting the counter lift.** Everything looks fine while all tenants stay continuously busy. As soon as an idle tenant returns, it takes every slot until its counter catches up, starving the active tenants for the duration of its absence.
- **Allowing the lift to lower a counter.** Assigning $\min\{c_i\}$ directly instead of taking the `max` with the tenant's existing counter resets a heavily served tenant whenever its queue briefly empties. A tenant sending requests one at a time can then game the scheduler.
- **Lifting only when the waiting queue is non-empty.** Skipping the lift when the queue is empty leaves a returning tenant with a stale counter. Using the counter of the most recently admitted tenant handles the empty-queue case.
- **Charging the entire request at admission.** Charging by a request's maximum token limit bills for tokens that may never be generated and treats different-length requests identically, drifting toward request counting. Charging by the true output length relies on future knowledge a server does not have.
- **Charging output tokens only at completion.** While a long generation runs, its tenant still looks under-served and keeps getting admitted ahead of others. The charge arrives only after the unfairness has already happened.
- **Counting requests instead of weighted tokens.** Equal turns do not mean equal service. In the three-tenant backlog test, request counting lets the service gap grow to 2,482 weighted tokens and keep widening over time, whereas VTC stays at 135.

## Complexity

`submit` and `_pick` both scan the waiting queue $Q$, taking $O(|Q|)$ time per arrival and per admission. Replacing the flat list with per-tenant FIFO queues and a min-heap of queued tenants keyed by counter reduces both operations to $O(\log T)$ for $T$ tenants. Counters change at most `max_running` times per step, so heap churn stays low. Memory overhead is one float per tenant.

## In the wild

- **The paper.** Sheng et al. ("Fairness in Serving Large Language Models", OSDI 2024) implement VTC in a vLLM-based server and compare it against first-come, first-served scheduling and per-tenant rate limits. The paper also covers production extensions such as per-tenant weights for paid tiers (dividing charges by each tenant's weight), alternative cost functions, and output-length prediction to tighten the fairness bound.
- **Networking roots.** VTC adapts start-time fair queueing from packet networks. The counter lift plays the same role as virtual time in network schedulers and wake-up placement in Linux's CFS, where a newly active flow starts at the current virtual time instead of zero.
- **Serving engines and gateways.** vLLM, TGI, and SGLang schedule requests first-come, first-served (with optional priority queues in vLLM) and leave multi-tenancy to an upstream gateway. Gateways typically enforce requests-per-minute or tokens-per-minute limits, which reject work even on idle GPUs, whereas VTC only reorders queued requests.
- **Limitations.** VTC controls admission only. Once admitted, a tenant with long-running sequences still occupies its slots (which requires preemption, as in `preemption-recompute`), and the fairness bound scales with batch size because a full batch of one tenant's requests must finish before counters rebalance.
