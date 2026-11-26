"""Tests for capstone-inference-gateway.

`Gateway` stands in front of the kit's `SimCluster`. The tests never compare whole
traces with the oracle: a gateway is judged the way a production one is, by rules
that must hold at every decision and by service-level numbers with margins.

`Watched`, a small subclass of `SimCluster` defined here, records the state of every
replica at the moment a request is handed to one, so each decision is checked
against exactly what the gateway could see: admission against a shadow token bucket
and the real queue, routing against the load (and, in stage 3, a shadow prefix
index), dispatch order against the queue, scaling against the policy. Goodput,
fairness, cache-hit tokens, and replica-seconds have limits derived from the
reference, stated in the statement.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, replace

from learnai.judge import case, check
from learnai.kit import CostModel, SimCluster, SimRequest, workloads

S1 = "1: admission control"
S2 = "2: least-loaded routing"
S3 = "3: prefix-aware routing"
S4 = "4: fair scheduling"
S5 = "5: autoscaling and draining"

COST = CostModel(base_s=2**-6, prefill_s_per_token=2**-12, decode_s_per_seq=2**-9)
QUANTUM = 2**-10  # arrival times are multiples of this
SLO_S = 1.0  # a good request has its first token within one second of arriving


@dataclass(frozen=True)
class Config:
    max_queued_requests: int = 10**6
    max_queued_tokens: int = 10**9
    replica_slots: int = 8
    rate_tokens_per_s: float | None = None
    burst_tokens: int = 0
    prefix_routing: bool = False
    match_threshold: float = 0.25
    imbalance_tokens: int = 0
    index_size: int = 3
    fair: bool = False
    w_in: float = 1.0
    w_out: float = 2.0
    period_s: float = 1.0
    target: int = 6
    tolerance: float = 0.25
    min_replicas: int = 1
    max_replicas: int = 40
    window_s: float = 20.0


@dataclass(frozen=True)
class Decision:
    time: float
    request: SimRequest
    rejected: object  # None, or the Rejected the gateway answered
    queued_requests: int  # in the gateway queue just before the call
    queued_tokens: int


@dataclass(frozen=True)
class Dispatch:
    time: float
    request: SimRequest
    replica: int
    state: str
    replicas: tuple  # (id, state, queue_len, outstanding_tokens) of every live replica
    queue: tuple  # ids in the gateway queue just before, oldest first


@dataclass(frozen=True)
class Tick:
    time: float
    metrics: object
    queued: int
    serving_after: int
    returned: object


class Watched(SimCluster):
    """A SimCluster that records what every replica looked like at each dispatch."""

    def __init__(self, sim, **options):
        self.dispatches = []
        self.queue = {}  # id -> request: admitted and not dispatched, oldest first
        self.log = []  # ("submit" | "dispatch" | "finish", request), as the gateway saw them
        self.last_event = 0.0  # time of the last arrival, completion, or evaluation
        super().__init__(sim, COST, **options)
        for replica in self.all_replicas:
            self._watch(replica)

    def add_replica(self):
        replica = super().add_replica()
        self._watch(replica)
        return replica

    def _watch(self, replica):
        inner = replica.submit

        def submit(request, **options):
            snapshot = tuple(
                (r.id, r.state, r.queue_len, r.outstanding_tokens) for r in self.replicas
            )
            self.dispatches.append(
                Dispatch(self.sim.now(), request, replica.id, replica.state, snapshot,
                         tuple(self.queue))
            )  # fmt: skip
            self.queue.pop(request.id, None)
            self.log.append(("dispatch", request))
            outcome = inner(request, **options)

            # Registered before the gateway's own callback, so it is logged first.
            def completed(done):
                self.last_event = self.sim.now()
                kind = "finish" if done.result().status == "finished" else "fail"
                self.log.append((kind, request))

            outcome.add_done_callback(completed)
            return outcome

        replica.submit = submit


class Run:
    def __init__(self, sim, cluster, gateway, requests, cfg, label):
        self.sim, self.cluster, self.gateway = sim, cluster, gateway
        self.requests, self.cfg, self.label = requests, cfg, label
        self.decisions = []
        self.ticks = []
        self.problems = []  # (message, detail), found while the simulation ran
        self.results = {}  # id -> the future submit returned
        self._summary = None

    def problem(self, message, detail=""):
        if not self.problems:
            self.problems.append((message, f"at t={self.sim.now():g} s: {detail}\n"))

    # -- measurements ----------------------------------------------------------------------

    def ttft(self, request):
        """Seconds from arrival to first token, or None if there was none."""
        outcome = self.cluster.outcomes.get(request.id)
        if outcome is None or outcome.status != "finished":
            return None
        return outcome.first_token_s - request.arrival_s

    def good(self, requests=None):
        """Requests that finished with their first token within the SLO."""
        chosen = self.requests if requests is None else requests
        return [r for r in chosen if (t := self.ttft(r)) is not None and t <= SLO_S]

    def rejected(self, reason=None):
        return [
            d.request
            for d in self.decisions
            if d.rejected is not None and (reason is None or d.rejected.reason == reason)
        ]

    def summary(self):
        """A few lines for failure details; computed once, after the simulation."""
        if self._summary is None:
            self._summary = self._summarize()
        return self._summary

    def _summarize(self):
        counters = self.cluster.counters
        tally = (
            f"rejected: {len(self.rejected())}, dispatched: {len(self.cluster.dispatches)}, "
            f"finished: {counters.finished}, failed: {counters.failed}, "
            f"good (first token within {SLO_S:g} s): {len(self.good())}"
        )
        lines = [f"{self.label}: {len(self.requests)} requests, cfg = {self.cfg}", tally]
        return "\n".join(lines) + "\n"


def simulate(ctx, module, requests, cfg, *, label="simulation", replicas=1, cache_tokens=0,
             autoscale=False, startup_s=4.0, until=None, probe_s=0.125):  # fmt: skip
    """Put `module.Gateway` in front of a watched cluster and play the workload."""
    sim = ctx.simclock()
    cluster = Watched(sim, replicas=replicas, max_seqs=8, cache_tokens=cache_tokens,
                      block_tokens=16, startup_s=startup_s)  # fmt: skip
    gateway = module.Gateway(cluster, cfg)
    run = Run(sim, cluster, gateway, requests, cfg, label)

    def arrive(request):
        before = (len(cluster.queue), sum(r.prompt_tokens for r in cluster.queue.values()))
        cluster.queue[request.id] = request  # visible to a dispatch made inside submit
        submit_at = len(cluster.log)
        result = gateway.submit(request)
        if not isinstance(result, asyncio.Future):
            cluster.queue.pop(request.id, None)
            run.problem("submit did not return an asyncio.Future", f"got {type(result).__name__}")
            return
        answer = result.result() if result.done() else None
        rejected = answer if isinstance(answer, module.Rejected) else None
        if rejected is not None:
            dispatched = any(d.request.id == request.id for d in cluster.dispatches)
            cluster.queue.pop(request.id, None)
            if dispatched:
                run.problem("a rejected request was sent to a replica", f"request {request.id!r}")
        else:
            cluster.log.insert(submit_at, ("submit", request))
            cluster.last_event = sim.now()
        run.results[request.id] = result
        run.decisions.append(Decision(sim.now(), request, rejected, *before))

    workloads.drive(sim, requests, arrive)
    horizon = until if until is not None else requests[-1].arrival_s + 120.0

    def probe():
        # Between events: no request may sit in the gateway while a ready replica has
        # room. The gateway dispatches when it admits a request, when a request
        # completes, and when it evaluates the autoscaling policy; a replica that became
        # ready after the last of those has not been offered work yet.
        if not cluster.queue:
            return
        room = [r.id for r in cluster.replicas
                if r.state == "ready" and r.queue_len < cfg.replica_slots]  # fmt: skip
        if room:
            ready_at = {e["replica"]: e.time for e in sim.trace.of("replica_ready")}
            room = [i for i in room if ready_at[i] < cluster.last_event]
        if room:
            run.problem(
                "a request waits in the gateway queue while a ready replica has a free slot",
                f"{len(cluster.queue)} queued; replicas with room: {room}",
            )

    for k in range(int(horizon / probe_s)):
        sim.call_at(k * probe_s + 2**-13, probe)

    if autoscale:

        def tick():
            # A plain timer with priority -1: an evaluation sees the cluster before the
            # arrivals and the step ends of its instant.
            metrics, queued = cluster.metrics(), len(cluster.queue)
            cluster.last_event = sim.now()
            n = gateway.evaluate()
            after = cluster.metrics()
            run.ticks.append(Tick(metrics.now, metrics, queued, after.ready + after.starting, n))

        for k in range(int(horizon / cfg.period_s) + 1):
            sim.call_at(k * cfg.period_s, tick, priority=-1)
    sim.run(until=horizon)
    return run


# -- rules that hold in every stage ----------------------------------------------------------


def check_rules(run, *, expect_all_served=True):
    detail = run.summary()
    cluster, cfg = run.cluster, run.cfg
    if run.problems:
        message, extra = run.problems[0]
        check.that(False, message, extra + detail)
    if cluster.violations:
        first = cluster.violations[0]
        check.that(False, f"the gateway misused the cluster ({first.code})",
                   f"{first}\n" + detail)  # fmt: skip
    for d in cluster.dispatches:
        me = next(row for row in d.replicas if row[0] == d.replica)
        check.that(
            d.state == "ready",
            "a request was sent to a replica that is not ready",
            f"t={d.time:g}: request {d.request.id!r} went to replica {d.replica}, "
            f"which was {d.state}\n" + detail,
        )
        check.that(
            me[2] < cfg.replica_slots,
            "a request was sent to a replica that had no free slot",
            f"t={d.time:g}: request {d.request.id!r} went to replica {d.replica}, which "
            f"held {me[2]} requests (replica_slots = {cfg.replica_slots})\n" + detail,
        )
    sent = [d.request.id for d in cluster.dispatches]
    check.that(len(sent) == len(set(sent)), "a request was sent to a replica twice", detail)
    admitted = [d.request for d in run.decisions if d.rejected is None]
    check.that(
        not cluster.counters.failed,
        f"{cluster.counters.failed} requests failed on a replica",
        detail + "".join(
            f"request {o.id!r}: {o.reason} on replica {o.replica}\n"
            for o in list(cluster.outcomes.values())
            if o.status == "failed"
        )[:1500],
    )  # fmt: skip
    if expect_all_served:
        unserved = [r.id for r in admitted if r.id not in cluster.outcomes]
        check.that(
            not unserved,
            f"{len(unserved)} admitted requests were never finished",
            detail + f"first: {unserved[:10]}\n",
        )
    for request in admitted:
        outcome, future = cluster.outcomes.get(request.id), run.results[request.id]
        if outcome is not None:
            check.that(
                future.done() and future.result() is outcome,
                "the future returned by submit did not resolve to the replica's Outcome",
                detail + f"request {request.id!r}\n",
            )


def check_admission(run):
    """Every admission decision, against a shadow token bucket and the real queue."""
    cfg, buckets = run.cfg, {}
    for d in run.decisions:
        request, cost = d.request, d.request.prompt_tokens
        expected = None
        balance = None
        if cfg.rate_tokens_per_s is not None:
            level, since = buckets.get(request.tenant, (cfg.burst_tokens, d.time))
            balance = min(cfg.burst_tokens, level + cfg.rate_tokens_per_s * (d.time - since))
            if balance < cost:
                retry = None
                if cost <= cfg.burst_tokens:
                    retry = (cost - balance) / cfg.rate_tokens_per_s
                expected = ("rate_limited", retry)
        if expected is None and (
            d.queued_requests + 1 > cfg.max_queued_requests
            or d.queued_tokens + cost > cfg.max_queued_tokens
        ):
            expected = ("queue_full", None)
        got = None if d.rejected is None else (d.rejected.reason, d.rejected.retry_after_s)
        where = (
            f"t={d.time:g}: request {request.id!r} of tenant {request.tenant!r} with "
            f"{cost} prompt tokens; the gateway queue held {d.queued_requests} requests "
            f"and {d.queued_tokens} tokens"
        )
        if balance is not None:
            where += f"; the tenant's bucket held {balance:g} tokens"
        same = got == expected or (
            got is not None
            and expected is not None
            and got[0] == expected[0]
            and got[1] is not None
            and expected[1] is not None
            and abs(got[1] - expected[1]) <= 1e-9
        )
        check.that(
            same,
            f"an admission decision is {show_answer(got)}, expected {show_answer(expected)}"
            if got is None or expected is None or got[0] != expected[0]
            else f"retry_after_s of a rejection is {got[1]!r}, expected {expected[1]!r}",
            f"{where}\nexpected: {show_answer(expected)}\ngot:      {show_answer(got)}\n"
            + run.summary(),
        )
        if d.rejected is not None:
            check.that(d.rejected.status == "rejected", "Rejected.status must be 'rejected'")
        if expected is None and balance is not None:
            buckets[request.tenant] = (balance - cost, d.time)


def show_answer(answer):
    if answer is None:
        return "admitted"
    return f"Rejected({answer[0]!r}, retry_after_s={answer[1]!r})"


def check_fcfs(run):
    """Without fair scheduling, requests leave the gateway queue oldest first."""
    for d in run.cluster.dispatches:
        check.that(
            d.queue and d.queue[0] == d.request.id,
            "the request sent to a replica was not the oldest in the gateway queue",
            f"t={d.time:g}: sent {d.request.id!r}; queue, oldest first: {list(d.queue[:8])}\n"
            + run.summary(),
        )


def check_least_loaded(run):
    """Stage 2: the ready replica with a free slot and the fewest outstanding tokens."""
    for d in run.cluster.dispatches:
        expected = least_loaded(d, run.cfg)
        check.that(
            d.replica == expected,
            "a request did not go to the least loaded replica with a free slot",
            f"t={d.time:g}: request {d.request.id!r} went to replica {d.replica}\n"
            + show_replicas(d) + run.summary(),
        )  # fmt: skip


def free_replicas(d, cfg):
    return [row for row in d.replicas if row[1] == "ready" and row[2] < cfg.replica_slots]


def least_loaded(d, cfg):
    return min(free_replicas(d, cfg), key=lambda row: (row[3], row[0]))[0]


def show_replicas(d):
    return "".join(
        f"  replica {i}: {state}, {queue_len} requests, {tokens} outstanding tokens\n"
        for i, state, queue_len, tokens in d.replicas
    )


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def check_prefix_routing(run):
    """Stage 3: every routing decision, against a shadow index of what went where."""
    cfg, index = run.cfg, {}

    def match(tokens, replica_id):
        return max((common_prefix(tokens, p) for p in index.get(replica_id, ())), default=0)

    for d in run.cluster.dispatches:
        tokens, free = d.request.tokens, free_replicas(d, cfg)
        if tokens is None:  # no prompt to match: routed by load, and not remembered
            check.that(d.replica == least_loaded(d, cfg),
                       "a request without token ids did not go to the least loaded replica",
                       show_replicas(d) + run.summary())  # fmt: skip
            continue
        least = min(row[3] for row in free)
        eligible = [row for row in free if row[3] - least <= cfg.imbalance_tokens]
        best = max(eligible, key=lambda row: (match(tokens, row[0]), -row[3], -row[0]))
        if match(tokens, best[0]) >= cfg.match_threshold * len(tokens):
            expected, why = best[0], f"the best eligible match ({match(tokens, best[0])} tokens)"
        else:
            expected, why = least_loaded(d, cfg), "no eligible match reaches the threshold"
        matches = ", ".join(f"replica {row[0]}: {match(tokens, row[0])}" for row in free)
        check.that(
            d.replica == expected,
            "a request did not go to the replica the routing rule chooses",
            f"t={d.time:g}: request {d.request.id!r} ({len(tokens)} prompt tokens) went to "
            f"replica {d.replica}\n" + show_replicas(d) + f"match with each free replica: "
            f"{matches}\nexpected: replica {expected} ({why})\n" + run.summary(),
        )
        index.setdefault(d.replica, deque(maxlen=cfg.index_size)).append(tokens)


def check_fair(run):
    """Stage 4: replay the gateway's own sequence of events through the VTC rules."""
    cfg, counters, queue, last = run.cfg, {}, [], None
    for kind, request in run.cluster.log:
        tenant = request.tenant
        if kind == "submit":
            counter = counters.setdefault(tenant, 0.0)
            queued = {r.tenant for r in queue}
            if tenant not in queued:
                if queued:
                    counter = max(counter, min(counters[t] for t in queued))
                elif last is not None:
                    counter = max(counter, counters[last])
                counters[tenant] = counter
            queue.append(request)
        elif kind == "dispatch":
            expected = min(queue, key=lambda r: counters[r.tenant])
            if request.id != expected.id:
                levels = ", ".join(f"{t!r}: {c:g}" for t, c in sorted(counters.items()))
                waiting = ", ".join(f"{r.id!r} ({r.tenant})" for r in queue[:10])
                check.that(
                    False,
                    f"request {request.id!r} was taken from the gateway queue, expected "
                    f"{expected.id!r} (the one fair scheduling chooses)",
                    f"request {request.id!r} of tenant {tenant!r} was sent to a replica\n"
                    f"counters at that moment: {levels}\nqueue, oldest first: {waiting}\n"
                    f"expected: request {expected.id!r} of tenant {expected.tenant!r}\n"
                    + run.summary(),
                )
            queue.remove(expected)
            counters[tenant] += cfg.w_in * request.prompt_tokens
            last = tenant
        elif kind == "finish":
            counters[tenant] += cfg.w_out * request.output_tokens
    got = getattr(run.gateway, "counters", None)
    check.that(
        isinstance(got, dict) and got == counters,
        "gateway.counters does not hold the specified virtual token counters at the end",
        f"expected: {counters}\ngot:      {got}\n" + run.summary(),
    )


def service_gap(run, tenants):
    """The largest drift between two tenants' service while all of them are queued.

    Service is w_in per prompt token when a request is sent to a replica and w_out
    per output token when it is generated. Returns max over pairs (f, g) and over
    intervals inside the backlogged period of |W_f - W_g|.
    """
    cfg, by_id = run.cfg, {r.id: r for r in run.requests}
    backlogged = [
        d.time for d in run.cluster.dispatches if {by_id[i].tenant for i in d.queue} >= set(tenants)
    ]
    if not backlogged:
        return 0.0
    start, end = backlogged[0], backlogged[-1]
    events = [(d.time, d.request.tenant, cfg.w_in * d.request.prompt_tokens)
              for d in run.cluster.dispatches]  # fmt: skip
    events += [(e.time, by_id[e["id"]].tenant, cfg.w_out) for e in run.sim.trace.of("token")]
    events.sort(key=lambda event: event[0])
    served = dict.fromkeys(tenants, 0.0)
    low, high = {}, {}
    for time, tenant, amount in events:
        if time > end:
            break
        if time >= start and tenant in served:
            served[tenant] += amount
        for f in tenants:
            for g in tenants:
                if f < g:
                    diff = served[f] - served[g]
                    low[f, g] = min(low.get((f, g), 0.0), diff)
                    high[f, g] = max(high.get((f, g), 0.0), diff)
    return max(high[pair] - low[pair] for pair in high)


def check_autoscale(run):
    """Stage 5: the replica count after every evaluation, against the policy."""
    cfg, history = run.cfg, []
    for tick in run.ticks:
        m = tick.metrics
        current = m.ready + m.starting
        load = tick.queued + m.queue_depth + m.running
        capacity = current * cfg.target
        raw = (
            current if abs(load - capacity) <= cfg.tolerance * capacity else -(-load // cfg.target)
        )
        raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))
        history = [(t, n) for t, n in history if m.now - t < cfg.window_s]
        history.append((m.now, raw))
        expected = raw if raw >= current else min(current, max(n for _, n in history))
        check.that(
            tick.returned == expected and type(tick.returned) is int,
            f"evaluate() returned {tick.returned!r}, expected {expected} (the policy's replica "
            "count)",
            f"evaluation at t={m.now:g}: ready={m.ready} starting={m.starting}, gateway "
            f"queue {tick.queued}, on replicas: {m.queue_depth} waiting + {m.running} "
            f"running (load {load})\nrecommendations in the window: "
            f"{[n for _, n in history]}\nexpected: {expected}\ngot:      {tick.returned!r}\n"
            + run.summary(),
        )
        check.that(
            tick.serving_after == expected,
            f"after an evaluation {tick.serving_after} replicas are starting or ready, expected "
            f"{expected} (the policy's count)",
            f"evaluation at t={m.now:g}: ready={m.ready} starting={m.starting} "
            f"draining={m.draining}, gateway queue {tick.queued}, on replicas: "
            f"{m.queue_depth} waiting + {m.running} running (load {load})\n"
            f"recommendations in the window: {[n for _, n in history]}\n"
            f"expected: {expected} replicas starting or ready\n"
            f"got:      {tick.serving_after}\n" + run.summary(),
        )


def merge(*groups):
    """Several tenants' requests as one workload, ids in arrival order."""
    ordered = sorted((r for group in groups for r in group), key=lambda r: r.arrival_s)
    return [replace(r, id=i) for i, r in enumerate(ordered)]


def tenant_requests(run, tenant):
    return [r for r in run.requests if r.tenant == tenant]


# -- stage 1: admission control --------------------------------------------------------------


def row(arrival, prompt, output, tenant):
    return (arrival, prompt, output, 0, tenant)


@case("the stage 1 example: a token bucket per tenant and a bounded queue", public=True,
      stage=S1)  # fmt: skip
def admission_example(ctx):
    cfg = Config(max_queued_requests=2, max_queued_tokens=100, replica_slots=1,
                 rate_tokens_per_s=64.0, burst_tokens=128)  # fmt: skip
    requests = workloads.replay([
        row(0.0, 64, 16, "a"), row(0.0, 48, 4, "a"), row(0.0, 32, 4, "a"), row(0.0, 40, 4, "b"),
        row(0.0, 40, 4, "b"), row(0.0, 8, 4, "b"), row(0.25, 32, 4, "a"), row(0.25, 200, 4, "c"),
        row(4.0, 128, 4, "a"), row(4.0, 1, 4, "a"),
    ])  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, cfg, label="the stage 1 example")
    check_rules(run)
    got = [None if d.rejected is None else (d.rejected.reason, d.rejected.retry_after_s)
           for d in run.decisions]  # fmt: skip
    expected = [
        None,  # a: 128 in the bucket, 64 left; straight to the idle replica
        None,  # a: 16 left; waits in the gateway queue (the replica's slot is taken)
        ("rate_limited", 0.25),  # a: 32 > 16; (32 - 16) / 64
        None,  # b: the queue holds 2 requests and 88 tokens
        ("queue_full", None),  # a third queued request
        ("queue_full", None),
        ("queue_full", None),  # a has 32 tokens again, but the queue is still full
        ("rate_limited", None),  # 200 tokens can never come out of a bucket of 128
        ("queue_full", None),  # an empty queue, but 128 > max_queued_tokens
        None,
    ]
    for k, (mine, want) in enumerate(zip(got, expected, strict=True)):
        check.that(
            mine == want,
            f"the stage 1 example: the answer to request {k} is {show_answer(mine)}, expected "
            f"{show_answer(want)}",
            f"expected: {show_answer(want)}\ngot:      {show_answer(mine)}\n" + run.summary(),
        )
    check_admission(run)
    check_fcfs(run)


def overload(ctx, n=600, rate=48.0):
    return workloads.poisson(ctx.rng, rate=rate, n=n, prompt_tokens=(16, 128),
                             output_tokens=(8, 24), tenants=("a", "b"), quantum_s=QUANTUM)  # fmt: skip


@case("overload: a bounded queue keeps the admitted requests within the SLO", public=True,
      stage=S1)  # fmt: skip
def bounded_queue(ctx):
    # Rejects unbounded_queue: everything is admitted and almost nothing is on time.
    cfg = Config(max_queued_requests=8, max_queued_tokens=1024)
    run = simulate(ctx, ctx.solution, overload(ctx), cfg, label="three times the capacity")
    check_rules(run)
    check_admission(run)
    check_fcfs(run)
    check.at_most("worst_queue_length", max(d.queued_requests + (d.rejected is None)
                                            for d in run.decisions), 8, unit="requests")  # fmt: skip
    ctx.metric("good_requests", len(run.good()), limit=130, higher_is_better=True)
    check.that(len(run.good()) >= 130, "too few requests had their first token within 1 s",
               run.summary())  # fmt: skip


@case("a tenant over its token rate is rejected; the others are not", public=True, stage=S1)
def rate_limit(ctx):
    bulk = workloads.poisson(ctx.rng, rate=30.0, n=300, prompt_tokens=(32, 128),
                             output_tokens=(4, 12), tenants=("bulk",), quantum_s=QUANTUM)  # fmt: skip
    chat = workloads.poisson(ctx.rng, rate=4.0, n=40, prompt_tokens=(8, 48),
                             output_tokens=(4, 12), tenants=("chat",), quantum_s=QUANTUM)  # fmt: skip
    cfg = Config(max_queued_requests=16, max_queued_tokens=4096, rate_tokens_per_s=256.0,
                 burst_tokens=512)  # fmt: skip
    run = simulate(ctx, ctx.solution, merge(bulk, chat), cfg, label="bulk and chat")
    check_rules(run)
    check_admission(run)
    chat_requests = tenant_requests(run, "chat")
    check.that(len(run.rejected("rate_limited")) >= 150, "the flooding tenant was not held "
               "to its rate", run.summary())  # fmt: skip
    check.that(len(run.good(chat_requests)) == len(chat_requests),
               "a request of the tenant within its rate was rejected or late", run.summary())  # fmt: skip


@case("the token bound of the queue, and requests that fit exactly", stage=S1)
def token_bound(ctx):
    # Rejects ignores_token_bound and running_counted_as_queued.
    cfg = Config(max_queued_requests=4, max_queued_tokens=200, replica_slots=2)
    requests = workloads.replay([
        row(0.0, 100, 40, "a"), row(0.0, 100, 40, "a"), row(0.0, 120, 8, "a"),
        row(0.0, 80, 8, "b"), row(0.0, 1, 8, "b"), row(0.001953125, 300, 8, "b"),
        row(0.00390625, 50, 8, "a"), row(3.0, 200, 8, "a"), row(3.0, 1, 8, "a"),
    ])  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, cfg, label="a queue of 200 tokens")
    check_rules(run)
    check_admission(run)
    check_fcfs(run)
    check.equal([d.rejected is None for d in run.decisions],
                [True, True, True, True, False, False, False, True, True],
                label="admitted (True) or rejected (False), in arrival order")  # fmt: skip


@case("buckets refill up to the burst, and an oversized prompt is never admitted", stage=S1)
def bucket_refill(ctx):
    # Rejects uncapped_bucket (the burst after the pause) and charges_rejected.
    cfg = Config(max_queued_requests=3, max_queued_tokens=10**6, replica_slots=1,
                 rate_tokens_per_s=128.0, burst_tokens=256)  # fmt: skip
    rows = [row(0.0, 100, 4, "a"), row(0.0, 100, 4, "a"), row(0.0, 100, 4, "a"),
            row(0.0, 300, 4, "b"), row(0.0, 60, 4, "b"), row(0.0, 60, 4, "b"),
            row(0.0, 60, 4, "b"), row(0.0, 60, 4, "b"), row(0.0, 60, 4, "b")]  # fmt: skip
    rows += [row(0.5 + k / 64, 40, 2, "a") for k in range(24)]
    rows += [row(20.0, 90, 2, t) for t in "aaabbb"] + [row(20.25, 33, 2, t) for t in "abab"]
    run = simulate(ctx, ctx.solution, workloads.replay(rows), cfg, label="bursts and pauses")
    check_rules(run)
    check_admission(run)


@case("random overload with both bounds and a rate limit: every decision is checked", stage=S1)
def admission_random(ctx):
    for slots, rate in ((8, 512.0), (3, None)):
        requests = workloads.poisson(ctx.rng, rate=40.0, n=300, prompt_tokens=(1, 200),
                                     output_tokens=(1, 24), tenants=("a", "b", "c"),
                                     quantum_s=QUANTUM)  # fmt: skip
        cfg = Config(max_queued_requests=6, max_queued_tokens=500, replica_slots=slots,
                     rate_tokens_per_s=rate, burst_tokens=600)  # fmt: skip
        run = simulate(ctx, ctx.solution, requests, cfg, label=f"{slots} slots")
        check_rules(run)
        check_admission(run)
        check_fcfs(run)


# -- stage 2: least-loaded routing ------------------------------------------------------------


@case("the stage 2 example: load is outstanding tokens, not requests", public=True, stage=S2)
def routing_example(ctx):
    # Rejects request_count_balancing (request 3) and ties_to_highest_id (request 0).
    requests = workloads.replay([
        row(0.0, 400, 60, "a"), row(0.0, 20, 4, "a"), row(0.0, 20, 4, "a"), row(0.0, 30, 4, "a"),
        row(0.0, 10, 4, "a"), row(0.0, 10, 4, "a"),
    ])  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, Config(), label="the stage 2 example",
                   replicas=3)  # fmt: skip
    check_rules(run)
    check.equal([d.replica for d in run.cluster.dispatches], [0, 1, 2, 1, 2, 2],
                label="replica of each request")  # fmt: skip
    check_least_loaded(run)


def mixed_sizes(ctx, n=400, rate=60.0, **options):
    """Mostly short requests and a few very long ones."""
    short = workloads.poisson(ctx.rng, rate=rate * 0.85, n=int(n * 0.85), prompt_tokens=(8, 64),
                              output_tokens=(2, 16), quantum_s=QUANTUM, **options)  # fmt: skip
    long = workloads.poisson(ctx.rng, rate=rate * 0.15, n=n - int(n * 0.85),
                             prompt_tokens=(300, 900), output_tokens=(40, 120),
                             quantum_s=QUANTUM, **options)  # fmt: skip
    return merge(short, long)


@case("four replicas, requests of very different sizes: every routing decision", public=True,
      stage=S2)  # fmt: skip
def routing_workload(ctx):
    run = simulate(ctx, ctx.solution, mixed_sizes(ctx), Config(), label="mixed sizes",
                   replicas=4)  # fmt: skip
    check_rules(run)
    check_least_loaded(run)
    check_fcfs(run)


@case("replicas with two slots: the rest waits in the gateway, oldest first", stage=S2)
def routing_with_queue(ctx):
    # Rejects overfills_replica.
    cfg = Config(replica_slots=2)
    run = simulate(ctx, ctx.solution, mixed_sizes(ctx, n=200, rate=40.0), cfg,
                   label="two slots per replica", replicas=3)  # fmt: skip
    check_rules(run)
    check_least_loaded(run)
    check_fcfs(run)
    check.that(max(d.queued_requests for d in run.decisions) > 4,
               "nothing ever waited in the gateway queue", run.summary())  # fmt: skip


@case("bursts onto idle replicas: ties and same-instant arrivals", stage=S2)
def routing_bursts(ctx):
    requests = workloads.bursty(ctx.rng, bursts=8, burst_size=14, period_s=6.0,
                                prompt_tokens=(8, 200), output_tokens=(2, 40))  # fmt: skip
    for replicas, slots in ((5, 8), (2, 3)):
        run = simulate(ctx, ctx.solution, requests, Config(replica_slots=slots),
                       label=f"bursts, {replicas} replicas", replicas=replicas)  # fmt: skip
        check_rules(run)
        check_least_loaded(run)
        check_fcfs(run)


# -- stage 3: prefix-aware routing -------------------------------------------------------------


def prompts(ctx, *, skew, rate, n=600):
    return workloads.shared_prefixes(ctx.rng, rate=rate, n=n, groups=8, prefix_tokens=(256, 512),
                                     suffix_tokens=(8, 48), skew=skew, output_tokens=(4, 32),
                                     quantum_s=QUANTUM)  # fmt: skip


def token_request(seq_id, arrival, tokens, output_tokens=4):
    return SimRequest(seq_id, arrival, len(tokens), output_tokens, tokens=tuple(tokens))


@case("the stage 3 example: follow a match, unless the replica is too far ahead", public=True,
      stage=S3)  # fmt: skip
def prefix_example(ctx):
    base = tuple(range(1, 41))  # 40 tokens
    other = tuple(range(101, 141))
    requests = [
        token_request(0, 0.0, base, 200),  # nothing remembered: least loaded, replica 0
        token_request(1, 0.0, (*base[:30], *[7] * 10), 20),  # 30 of 40 match replica 0
        token_request(2, 0.0, (*base[:8], *[9] * 32), 20),  # replica 0 is 300 ahead
        token_request(3, 0.0, other, 20),
        token_request(4, 0.0, (*base[:35], *[5] * 5), 20),  # 35 match; replica 0 is 240 ahead
        token_request(5, 0.0, (*base[:35], 6), 4),  # replica 0 is 300 ahead: not eligible
        token_request(6, 0.0, (*base[:35], 8), 4),  # replica 1 now remembers request 5
    ]
    cfg = Config(prefix_routing=True, match_threshold=0.5, imbalance_tokens=250, index_size=2)
    run = simulate(ctx, ctx.solution, requests, cfg, label="the stage 3 example", replicas=3,
                   cache_tokens=1200)  # fmt: skip
    check_rules(run)
    check.equal([d.replica for d in run.cluster.dispatches], [0, 0, 1, 2, 0, 1, 1],
                label="replica of each request")  # fmt: skip
    check_prefix_routing(run)


@case("eight prefixes on four replicas: more cache hits and a faster first token", public=True,
      stage=S3)  # fmt: skip
def prefix_workload(ctx):
    # Rejects index_not_updated and a gateway that still routes by load alone.
    cfg = Config(prefix_routing=True, match_threshold=0.25, imbalance_tokens=1500, index_size=3)
    # Routing by load alone hits 54144 cached tokens here and 433 requests are on time.
    run = simulate(ctx, ctx.solution, prompts(ctx, skew=0.0, rate=30.0), cfg,
                   label="eight prefixes", replicas=4, cache_tokens=1200)  # fmt: skip
    check_rules(run)
    check_prefix_routing(run)
    check.cost(run.cluster.counters, "cache_hit_tokens", at_least=90_000, unit="tokens")
    ctx.metric("good_requests", len(run.good()), limit=570, higher_is_better=True)
    check.that(len(run.good()) >= 570, "too few requests had their first token within 1 s",
               run.summary())  # fmt: skip


@case("one hot prefix: affinity is broken when its replica is too far ahead", stage=S3)
def hot_prefix(ctx):
    # Rejects sticky_routing.
    cfg = Config(prefix_routing=True, match_threshold=0.25, imbalance_tokens=400, index_size=3)
    run = simulate(ctx, ctx.solution, prompts(ctx, skew=2.0, rate=40.0, n=400), cfg,
                   label="a hot prefix", replicas=4, cache_tokens=1200)  # fmt: skip
    check_rules(run)
    check_prefix_routing(run)
    worst = max(
        next(r for r in d.replicas if r[0] == d.replica)[3]
        - min(r[3] for r in free_replicas(d, cfg))
        for d in run.cluster.dispatches
    )
    check.at_most("largest_imbalance_followed", worst, 400, unit="tokens")


@case("a short index, a strict threshold, and requests without token ids", stage=S3)
def prefix_variants(ctx):
    # Rejects index_unbounded.
    with_ids = prompts(ctx, skew=1.0, rate=30.0, n=200)
    plain = workloads.poisson(ctx.rng, rate=6.0, n=40, prompt_tokens=(8, 200),
                              output_tokens=(2, 24), quantum_s=QUANTUM)  # fmt: skip
    requests = merge(with_ids, plain)
    for threshold, imbalance, size, slots in ((0.75, 800, 1, 8), (0.0, 0, 2, 8), (1.0, 2000, 4, 3)):
        cfg = Config(prefix_routing=True, match_threshold=threshold, imbalance_tokens=imbalance,
                     index_size=size, replica_slots=slots)  # fmt: skip
        run = simulate(ctx, ctx.solution, requests, cfg, label=f"threshold {threshold}",
                       replicas=3, cache_tokens=1200)  # fmt: skip
        check_rules(run)
        check_prefix_routing(run)
        check_fcfs(run)
    off = simulate(ctx, ctx.solution, requests, Config(), label="prefix routing off", replicas=3,
                   cache_tokens=1200)  # fmt: skip
    check_least_loaded(off)


# -- stage 4: fair scheduling --------------------------------------------------------------------


@case("the stage 4 example: two tenants, one slot", public=True, stage=S4)
def fair_example(ctx):
    cfg = Config(fair=True, replica_slots=1)
    requests = workloads.replay([
        row(0.0, 64, 8, "a"), row(0.0, 64, 8, "a"), row(0.0, 64, 8, "a"), row(0.03125, 32, 4, "b"),
        row(0.03125, 32, 4, "b"), row(2.0, 64, 8, "a"), row(2.0, 32, 4, "b"), row(2.0, 32, 4, "b"),
    ])  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, cfg, label="the stage 4 example")
    check_rules(run)
    check.equal([d.request.id for d in run.cluster.dispatches], [0, 3, 1, 4, 2, 5, 6, 7],
                label="order in which requests left the gateway queue")  # fmt: skip
    check.equal(run.gateway.counters, {"a": 320.0, "b": 384.0}, label="counters at the end")
    check_fair(run)


def batch_and_chat(ctx):
    batch = workloads.poisson(ctx.rng, rate=60.0, n=300, prompt_tokens=(64, 256),
                              output_tokens=(16, 48), tenants=("batch",), quantum_s=QUANTUM)  # fmt: skip
    chat = workloads.poisson(ctx.rng, rate=4.0, n=60, prompt_tokens=(8, 64),
                             output_tokens=(4, 24), tenants=("chat",), quantum_s=QUANTUM)  # fmt: skip
    return merge(batch, chat)


@case("a batch tenant floods the queue: the interactive tenant is still served", public=True,
      stage=S4)  # fmt: skip
def no_starvation(ctx):
    # Rejects tenant_starvation: first come, first served leaves chat behind the backlog.
    run = simulate(ctx, ctx.solution, batch_and_chat(ctx), Config(fair=True),
                   label="batch and chat", replicas=2)  # fmt: skip
    check_rules(run)
    chat = tenant_requests(run, "chat")
    ctx.metric("good_chat_requests", len(run.good(chat)), limit=54, higher_is_better=True)
    check.that(len(run.good(chat)) >= 54, "too few requests of the interactive tenant had "
               "their first token within 1 s", run.summary())  # fmt: skip
    check_fair(run)


def three_tenants(ctx):
    a = workloads.poisson(ctx.rng, rate=40.0, n=150, prompt_tokens=(8, 32),
                          output_tokens=(2, 8), tenants=("a",), quantum_s=QUANTUM)  # fmt: skip
    b = workloads.poisson(ctx.rng, rate=20.0, n=80, prompt_tokens=(100, 300),
                          output_tokens=(8, 16), tenants=("b",), quantum_s=QUANTUM)  # fmt: skip
    c = workloads.poisson(ctx.rng, rate=20.0, n=80, prompt_tokens=(16, 64),
                          output_tokens=(24, 64), tenants=("c",), quantum_s=QUANTUM)  # fmt: skip
    return merge(a, b, c)


@case("three backlogged tenants with different request shapes get equal service", stage=S4)
def equal_service(ctx):
    # Rejects counts_requests and output_not_charged.
    run = simulate(ctx, ctx.solution, three_tenants(ctx), Config(fair=True),
                   label="three tenants", replicas=2)  # fmt: skip
    check_rules(run)
    gap = service_gap(run, ("a", "b", "c"))
    check.at_most("service_gap", gap, 2000.0, unit="weighted tokens")
    check_fair(run)


@case("tenants that come and go: an idle tenant does not save up credit", stage=S4)
def comes_and_goes(ctx):
    # Rejects no_lift.
    steady = workloads.poisson(ctx.rng, rate=14.0, n=260, prompt_tokens=(16, 96),
                               output_tokens=(4, 16), tenants=("steady",), quantum_s=QUANTUM)  # fmt: skip
    early = workloads.poisson(ctx.rng, rate=10.0, n=20, prompt_tokens=(16, 96),
                              output_tokens=(4, 16), tenants=("visitor",), quantum_s=QUANTUM)  # fmt: skip
    late = workloads.poisson(ctx.rng, rate=40.0, n=80, prompt_tokens=(16, 96),
                             output_tokens=(4, 16), tenants=("visitor",), quantum_s=QUANTUM,
                             start_s=12.0)  # fmt: skip
    requests = merge(steady, early, late)
    for slots, weights in ((1, (1.0, 2.0)), (4, (0.5, 4.0))):
        cfg = Config(fair=True, replica_slots=slots, w_in=weights[0], w_out=weights[1])
        run = simulate(ctx, ctx.solution, requests, cfg, label=f"{slots} slots")
        check_rules(run)
        check_fair(run)
    off = simulate(ctx, ctx.solution, requests, Config(replica_slots=4), label="fair off")
    check_fcfs(off)


@case("fair scheduling behind admission control and in front of prefix routing", stage=S4)
def fair_with_everything(ctx):
    a = workloads.shared_prefixes(ctx.rng, rate=30.0, n=200, groups=4, prefix_tokens=(128, 256),
                                  tenants=("a",), quantum_s=QUANTUM)  # fmt: skip
    b = workloads.shared_prefixes(ctx.rng, rate=10.0, n=70, groups=2, prefix_tokens=(128, 256),
                                  tenants=("b", "c"), quantum_s=QUANTUM)  # fmt: skip
    cfg = Config(fair=True, max_queued_requests=24, max_queued_tokens=6000,
                 rate_tokens_per_s=2048.0, burst_tokens=4096, prefix_routing=True,
                 imbalance_tokens=600, replica_slots=4)  # fmt: skip
    run = simulate(ctx, ctx.solution, merge(a, b), cfg, label="all of stages 1 to 4",
                   replicas=2, cache_tokens=1200)  # fmt: skip
    check_rules(run)
    check_admission(run)
    check_prefix_routing(run)
    check_fair(run)


# -- stage 5: autoscaling and draining -------------------------------------------------------------


def scaled(ctx, phases, cfg, label, *, until, replicas=1):
    requests = workloads.phases(ctx.rng, phases, prompt_tokens=(8, 64), output_tokens=(4, 32),
                                quantum_s=QUANTUM)  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, cfg, label=label, replicas=replicas,
                   autoscale=True, until=until)  # fmt: skip
    check_rules(run)
    check_autoscale(run)
    return run


def report_scaling(ctx, run, *, min_good, max_replica_seconds):
    good, paid = len(run.good()), run.cluster.replica_seconds()
    ctx.metric("good_requests", good, limit=min_good, higher_is_better=True)
    check.that(good >= min_good, "too few requests had their first token within 1 s",
               run.summary())  # fmt: skip
    check.at_most("replica_seconds", paid, max_replica_seconds, unit="s")


@case("a load step: scale up through the start-up delay, scale down late", public=True, stage=S5)
def load_step(ctx):
    run = scaled(ctx, [(20.0, 5.0), (40.0, 30.0), (40.0, 5.0)], Config(), "a load step",
                 until=120.0)  # fmt: skip
    report_scaling(ctx, run, min_good=1300, max_replica_seconds=650)


@case("bursts 6 s apart: the replicas survive the gaps", stage=S5)
def burst_load(ctx):
    phases = [(10.0, 5.0)] + [(6.0, 30.0), (6.0, 5.0)] * 6 + [(30.0, 5.0)]
    run = scaled(ctx, phases, Config(), "bursts", until=132.0)
    report_scaling(ctx, run, min_good=1200, max_replica_seconds=700)


@case("scaling down while requests are running: nothing fails", stage=S5)
def drain_keeps_requests(ctx):
    # Rejects kills_in_flight. A short window, long answers: replicas are removed
    # while they are full.
    cfg = Config(window_s=2.0, target=4, min_replicas=1, max_replicas=12)
    requests = workloads.phases(ctx.rng, [(6.0, 40.0), (20.0, 2.0)], prompt_tokens=(8, 64),
                                output_tokens=(100, 300), quantum_s=QUANTUM)  # fmt: skip
    run = simulate(ctx, ctx.solution, requests, cfg, label="scale down under load", replicas=8,
                   autoscale=True, startup_s=1.0, until=80.0)  # fmt: skip
    check_rules(run)
    check_autoscale(run)
    check.that(run.cluster.counters.scale_downs >= 3, "the cluster was never scaled down",
               run.summary())  # fmt: skip


@case("the whole gateway: limits, prefixes, fairness, and scaling together", stage=S5)
def whole_gateway(ctx):
    a = workloads.shared_prefixes(ctx.rng, rate=25.0, n=700, groups=6, prefix_tokens=(128, 384),
                                  tenants=("a",), quantum_s=QUANTUM, start_s=5.0)  # fmt: skip
    b = workloads.shared_prefixes(ctx.rng, rate=5.0, n=250, groups=3, prefix_tokens=(128, 384),
                                  tenants=("b", "c"), quantum_s=QUANTUM)  # fmt: skip
    cfg = Config(fair=True, max_queued_requests=40, max_queued_tokens=12000,
                 rate_tokens_per_s=6000.0, burst_tokens=12000, prefix_routing=True,
                 imbalance_tokens=800, window_s=8.0)  # fmt: skip
    run = simulate(ctx, ctx.solution, merge(a, b), cfg, label="the whole gateway", replicas=1,
                   cache_tokens=1200, autoscale=True, until=90.0)  # fmt: skip
    check_rules(run)
    check_admission(run)
    check_prefix_routing(run)
    check_fair(run)
    check_autoscale(run)
    report_scaling(ctx, run, min_good=700, max_replica_seconds=320)
