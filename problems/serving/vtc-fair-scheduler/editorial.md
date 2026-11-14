## The idea

Fair queueing gives each backlogged tenant the same amount of *service*, however the
tenants split it into requests. For an LLM server, the paper measures service in
weighted tokens, $w_\text{in}$ per prompt token and $w_\text{out}$ per generated
token (1 and 2 by default, following API pricing), and keeps one number per tenant:
the virtual token counter $c_i$, the service tenant $i$ has been charged so far.

**Serve the smallest counter.** When a slot is free, the queued tenant with the
smallest counter has received the least and goes next. Its counter rises as it is
served, so it stops being the smallest, and the counters of queued tenants chase each
other. That is the entire scheduling rule.

**Charge when served, at token granularity.** The prompt is charged at admission,
immediately, so that the remaining picks of the same step see it and one tenant does
not take every free slot. Output tokens are charged one by one as they come out of
the engine. This is how VTC copes with not knowing output lengths: it never needs an
estimate, because the counter is corrected after every step, and a tenant in the
middle of a long generation already looks as served as it is.

**Lift on return.** A counter that only ever grows remembers idleness as credit: a
tenant that sent nothing for an hour has a counter far below everyone else's and
would monopolize the engine until it caught up. Fairness in the paper's sense is about
the present, among tenants that are competing *now*; capacity a tenant did not use is
gone. So when a tenant becomes queued again, its counter is raised to the smallest
counter among the queued tenants: it joins the back of the pack, not the front. If
nobody is queued, the reference point is the tenant served last. The `max` makes the
lift one-directional: a tenant that is *ahead* (heavily served, briefly out of
requests) keeps its counter, otherwise emptying one's queue would erase one's debt.

With these rules the counters of queued tenants never differ by more than
$U = \max(w_\text{in} L_\text{input},\ w_\text{out} M)$ (Lemma 4.3), and since a
backlogged tenant's service over an interval is exactly the growth of its counter,
two backlogged tenants differ by at most $2U$ over any interval (Theorem 4.4). The
bound does not grow with the length of the interval, which is the point.

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

`min` over the waiting requests, keyed by their tenant's counter, returns the first
request in queue order among those of the least-served tenants: the tie rule and
"earliest request of that tenant" in one expression.

## Common mistakes

- **No lift.** Everything works while all tenants are busy. The first tenant that
  returns from a quiet period takes every slot until its counter has caught up, and
  the tenants that were working are starved for exactly as long as they had been
  productive.
- **A lift that can lower.** Assigning $\min\{c_i\}$ instead of taking the maximum
  with the tenant's own counter resets a heavily served tenant every time its queue
  runs empty for a moment. Sending requests one at a time then beats the scheduler.
- **Lifting only against queued tenants.** When the queue is empty there is nobody to
  take the minimum over, and skipping the lift leaves the newcomer with its stale
  counter. The paper's answer is the counter of the tenant that was served last.
- **Charging the whole request at admission.** Using the request's token limit or its
  eventual length makes the counter jump at admission and stand still during
  generation. With a limit, the charge is for tokens that may never be generated, and
  it is the same for every request, so the scheduler drifts toward counting requests;
  with the true length, it needs information a server does not have.
- **Charging output at completion.** While a long answer is being generated its
  tenant looks under-served and is admitted again and again; the bill arrives after
  the damage.
- **Counting requests.** Equal turns are not equal service. In the three-tenant
  backlog of the tests, the service difference grows to 2,482 weighted tokens and
  keeps growing with the length of the run; VTC stays at 135.

## Complexity

`submit` and `_pick` scan the waiting list, $O(|Q|)$ per arrival and per admission.
With per-tenant FIFOs and a heap of queued tenants keyed by counter, both become
$O(\log T)$ for $T$ tenants; a counter changes at most `max_running` times per step,
so the heap sees little churn. Memory is one float per tenant.

## In the wild

- **The paper** (Sheng, Cao, Li, Zhu, et al., "Fairness in Serving Large Language
  Models", OSDI 2024) implements VTC in a vLLM-based server and compares it with
  first come, first served and with requests-per-minute limits. It also gives the
  variants a production system wants: per-tenant weights (divide the charge by the
  tenant's weight, for paid tiers), other cost functions than a weighted token count,
  and length prediction to tighten the bound.
- **Where it comes from.** VTC is start-time fair queueing from packet networks, and
  the lift is the "virtual time" of those schedulers: a flow that becomes active
  starts at the current virtual time, not at zero. Linux's CFS does the same when a
  sleeping task wakes up, for the same reason.
- **Engines.** vLLM, TGI, and SGLang schedule first come, first served (vLLM also by
  priority) and leave tenancy to the layer in front. Gateways mostly enforce
  per-tenant rate limits in requests or tokens per minute, which is the baseline VTC
  improves on: a limit rejects work even when the GPUs are idle, a fair queue only
  reorders it.
- **Limits.** Fairness is about admission here; a tenant with long-running sequences
  still holds its slots (`preemption-recompute` is the tool for that), and the bound
  scales with the batch size, since a full batch of one tenant's requests must finish
  before the counters can even out.
