"""Tests for vtc-fair-scheduler.

`simulate` calls `scheduler.submit(request)` at every arrival and runs
`scheduler.run()` against a `SimEngine` on a simulated clock whose unit is the
millisecond. The statement determines every admission, so the cases check the
admissions, the engine steps, and the counters: against literal values on the
hand-checked scenarios, and on seeded workloads against the oracle and against
properties computed here from the trace (the service-difference bound of the VTC
paper, work conservation, first come first served within a tenant).
"""

from itertools import combinations

from learnai.judge import case, check
from learnai.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, scheduler, requests, max_running, w_in, w_out):
        self.sim = sim
        self.engine = engine
        self.scheduler = scheduler
        self.requests = requests
        self.by_id = {r.id: r for r in requests}
        self.max_running = max_running
        self.w_in = w_in
        self.w_out = w_out

    @property
    def steps(self):
        return [(e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")]

    @property
    def admissions(self):
        """[(time, id)] in admission order."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("admit")]

    @property
    def snapshots(self):
        """{time: counters} for the instants the case asked for."""
        return {e.time: e["counters"] for e in self.sim.trace.of("snapshot")}

    def final_counters(self):
        return {t: float(c) for t, c in self.scheduler.counters.items()}

    def timeline(self):
        settings = f"max_running={self.max_running}, w_in={self.w_in}, w_out={self.w_out}"
        lines = [f"{settings}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        shown = [e for e in self.sim.trace.events if e.kind in ("arrive", "admit", "finish")]
        for e in shown[:80]:
            r = self.by_id[e["id"]]
            what = f"{e.kind:<7} {r.id!r} (tenant {r.tenant}"
            if e.kind == "arrive":
                what += f", prompt {r.prompt_tokens}, output {r.output_tokens}"
            lines.append(f"  t={e.time:>8g}  {what})")
        if len(shown) > 80:
            lines.append(f"  ... {len(shown) - 80} more events")
        return "\n".join(lines) + "\n"


def horizon(requests):
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    return requests[-1].arrival_s + 2 * serial + 1000


def simulate(ctx, scheduler_class, requests, *, max_running, w_in=1.0, w_out=2.0, snapshot_at=()):
    """Run a scheduler over `requests`. `snapshot_at` lists instants at which a copy of
    `scheduler.counters` is recorded; use instants at which no step ends."""
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    scheduler = scheduler_class(engine, max_running, w_in, w_out)
    run = Run(sim, engine, scheduler, requests, max_running, w_in, w_out)
    workloads.drive(sim, requests, scheduler.submit)
    for when in snapshot_at:
        sim.call_at(when, lambda: sim.trace.record("snapshot", counters=dict(scheduler.counters)))
    sim.spawn(scheduler.run())
    try:
        sim.run(until=horizon(requests))
    except ProtocolViolation:
        pass  # reported below, with the timeline
    if engine.violations:
        first = engine.violations[0]
        check.that(
            False,
            f"the scheduler broke the engine's protocol ({first.code})",
            f"{first.code}: {first}\n" + run.timeline(),
        )
    finished = {e["id"] for e in sim.trace.of("finish")}
    missing = [r.id for r in requests if r.id not in finished]
    check.that(
        not missing,
        f"{len(missing)} of {len(requests)} requests never finished",
        f"unfinished: {missing[:10]!r}\n" + run.timeline(),
    )
    check.that(
        isinstance(scheduler.counters, dict),
        "scheduler.counters must be a dict from tenant to counter",
    )
    return run


def solve(ctx, requests, **options):
    return simulate(ctx, ctx.solution.VTCScheduler, requests, **options)


def expect_admissions(run, expected):
    got = run.admissions
    detail = f"expected admissions (ms, id): {expected[:60]!r}\n"
    detail += f"got admissions      (ms, id): {got[:60]!r}\n" + run.timeline()
    for k, (have, want) in enumerate(zip(got, expected, strict=False), start=1):
        check.that(
            have[1] == want[1],
            f"admission {k} is request {have[1]!r}, expected request {want[1]!r}",
            detail,
        )
        check.that(
            have[0] == want[0],
            f"admission {k} happens at {have[0]!r} ms, expected {want[0]!r}",
            detail,
        )
    check.that(
        len(got) == len(expected),
        f"{len(got)} requests were admitted, expected {len(expected)}",
        detail,
    )


def expect_counters(got, want, what, run):
    """Compare a {tenant: counter} dict exactly."""
    detail = f"expected {what}: {want!r}\ngot: {got!r}\n" + run.timeline()
    check.that(
        set(got) == set(want),
        f"{what}: the tenants are {sorted(got, key=str)!r}, expected {sorted(want, key=str)!r}",
        detail,
    )
    bad = next((t for t in want if float(got[t]) != float(want[t])), None)
    check.that(
        all(float(got[t]) == float(want[t]) for t in want),
        f"{what}: the virtual token counter of tenant {bad!r} is {got.get(bad)!r}, expected "
        f"{want.get(bad)!r}",
        detail,
    )


def service_events(run):
    """[(time, tenant, weighted service)]: input tokens when a request is admitted,
    each output token when it is produced. This is the paper's service measure."""
    events = []
    for e in run.sim.trace.events:
        if e.kind == "admit":
            r = run.by_id[e["id"]]
            events.append((e.time, r.tenant, run.w_in * r.prompt_tokens))
        elif e.kind == "token":
            events.append((e.time, run.by_id[e["id"]].tenant, run.w_out))
    return events


def backlog_intervals(run):
    """{tenant: [(t1, t2)]}: maximal intervals [t1, t2) during which the tenant has at
    least one request waiting (arrived and not admitted)."""
    waiting, since, intervals = {}, {}, {}
    for e in run.sim.trace.events:
        if e.kind not in ("arrive", "admit"):
            continue
        tenant = run.by_id[e["id"]].tenant
        count = waiting.get(tenant, 0)
        if e.kind == "arrive":
            if count == 0:
                since[tenant] = e.time
            waiting[tenant] = count + 1
        else:
            waiting[tenant] = count - 1
            if count == 1 and e.time > since[tenant]:
                intervals.setdefault(tenant, []).append((since[tenant], e.time))
    return intervals


def largest_service_difference(run):
    """max |W_f(s1, s2) - W_g(s1, s2)| over tenants f, g and intervals [s1, s2) in which
    both are backlogged; also returns where it happens."""
    events = service_events(run)
    intervals = backlog_intervals(run)
    worst, where = 0.0, None
    for f, g in combinations(sorted(intervals), 2):
        for a1, a2 in intervals[f]:
            for b1, b2 in intervals[g]:
                t1, t2 = max(a1, b1), min(a2, b2)
                if t1 >= t2:
                    continue
                # Prefix sums of W_f - W_g at every instant of [t1, t2]; the largest
                # difference over sub-intervals is their range.
                acc, low, high, last = 0.0, 0.0, 0.0, None
                for when, tenant, amount in events:
                    if not t1 <= when < t2 or tenant not in (f, g):
                        continue
                    if when != last:
                        low, high, last = min(low, acc), max(high, acc), when
                    acc += amount if tenant == f else -amount
                low, high = min(low, acc), max(high, acc)
                if high - low > worst:
                    worst, where = high - low, (f, g, t1, t2)
    return worst, where


def paper_bound(run):
    """2 * max(w_in * L_input, w_out * M): Theorem 4.4 of the VTC paper. M is the number
    of tokens that fit in a running batch: max_running sequences of L_input + L_output."""
    l_in = max(r.prompt_tokens for r in run.requests)
    l_out = max(r.output_tokens for r in run.requests)
    return 2 * max(run.w_in * l_in, run.w_out * run.max_running * (l_in + l_out))


def check_fairness(ctx, run, name="service_difference"):
    worst, where = largest_service_difference(run)
    bound = paper_bound(run)
    ctx.metric(name, worst, unit="weighted tokens", limit=bound)
    check.that(
        worst <= bound,
        "two backlogged tenants received service that differs by more than the bound "
        "2 * max(w_in * L_input, w_out * M)",
        f"difference {worst:g} > bound {bound:g} for tenants {where[0]!r} and {where[1]!r} "
        f"within [{where[2]:g}, {where[3]:g})\n" + run.timeline() if where else None,
    )


def check_discipline(run):
    """Work conservation and first come, first served within each tenant."""
    detail = run.timeline()
    order = {}
    for _, seq_id in run.admissions:
        order.setdefault(run.by_id[seq_id].tenant, []).append(seq_id)
    for tenant, admitted in order.items():
        arrived = [r.id for r in run.requests if r.tenant == tenant]
        check.that(
            admitted == arrived,
            "the requests of one tenant were not admitted in arrival order",
            f"tenant {tenant!r}: arrived {arrived[:20]!r}, admitted {admitted[:20]!r}\n" + detail,
        )
    admitted_at = {seq_id: when for when, seq_id in run.admissions}
    finish = {e["id"]: e.time for e in run.sim.trace.of("finish")}
    for when, _, _ in run.steps:
        live = sum(1 for i, t in admitted_at.items() if t <= when and finish[i] > when)
        waiting = any(r.arrival_s <= when and admitted_at[r.id] > when for r in run.requests)
        check.that(
            not waiting or live == run.max_running,
            "a slot was free at the start of a step while a request was waiting",
            f"step at t={when:g}: {live} live sequences\n" + detail,
        )


def matches_oracle(ctx, requests, **options):
    got = solve(ctx, requests, **options)
    check_discipline(got)
    want = simulate(ctx, ctx.oracle.VTCScheduler, requests, **options)
    expect_admissions(got, want.admissions)
    k = next((k for k, (a, b) in enumerate(zip(got.steps, want.steps, strict=False)) if a != b), 0)
    check.that(
        got.steps == want.steps,
        "the admissions are right but the engine steps differ from the continuous-batching loop",
        got.timeline() + f"first difference: step {k}, expected {want.steps[k : k + 1]!r}\n",
    )
    expect_counters(got.final_counters(), want.final_counters(), "final counters", got)
    return got


def req(id, arrival, tenant, prompt=8, output=2):
    return SimRequest(id, arrival, prompt, output, tenant=tenant)


# The worked example. With one slot, a request (prompt 8, output 2) holds the slot for
# 19 ms and costs its tenant 1 * 8 + 2 * 2 = 12.
EXAMPLE = [req(0, 0, "a"), req(1, 0, "a"), req(2, 0, "a"), req(3, 5, "b"),
           req(4, 60, "b"), req(5, 60, "b"), req(6, 61, "a"), req(7, 61, "a")]  # fmt: skip


@case("the worked example: least-served tenant first, and a lift for a returning tenant",
      public=True)
def worked_example(ctx):
    run = solve(ctx, EXAMPLE, max_running=1, snapshot_at=(5.5, 60.5, 61.5))
    expect_admissions(run, [(0, 0), (19, 3), (38, 1), (57, 2), (76, 4), (95, 6), (114, 5),
                            (133, 7)])  # fmt: skip
    snapshots = run.snapshots
    expect_counters(snapshots[5.5], {"a": 8, "b": 8}, "counters at t=5.5", run)
    expect_counters(snapshots[60.5], {"a": 32, "b": 32}, "counters at t=60.5", run)
    expect_counters(snapshots[61.5], {"a": 32, "b": 32}, "counters at t=61.5", run)
    expect_counters(run.final_counters(), {"a": 60, "b": 56}, "final counters", run)


@case("tokens are charged as they are served: input on admission, output one by one",
      public=True)
def charged_as_served(ctx):
    # Rejects charges_at_admission, output_charged_at_finish, and counts_requests.
    # One request, prompt 40, output 4: admitted at 0, tokens at 18, 27, 36, 45.
    run = solve(ctx, [SimRequest("r", 0, 40, 4, tenant="t")], max_running=2,
                snapshot_at=(0.5, 18.5, 27.5, 36.5, 45.5))  # fmt: skip
    for when, want in ((0.5, 40), (18.5, 42), (27.5, 44), (36.5, 46), (45.5, 48)):
        expect_counters(run.snapshots[when], {"t": want}, f"counters at t={when}", run)
    # Other weights: 3 per input token, 0.5 per output token.
    run = solve(ctx, [SimRequest("r", 0, 40, 4, tenant="t")], max_running=2, w_in=3.0, w_out=0.5,
                snapshot_at=(0.5, 27.5))  # fmt: skip
    expect_counters(run.snapshots[0.5], {"t": 120}, "counters at t=0.5", run)
    expect_counters(run.snapshots[27.5], {"t": 121}, "counters at t=27.5", run)
    expect_counters(run.final_counters(), {"t": 122}, "final counters", run)


@case("two backlogged tenants with different request sizes get equal service, not "
      "equal turns", public=True)  # fmt: skip
def equal_service_not_equal_turns(ctx):
    # Rejects counts_requests: "big" sends requests that cost 40 + 2 * 6 = 52, "small"
    # requests that cost 8 + 2 * 2 = 12, so "small" is admitted about four times as often.
    requests = [SimRequest(f"big-{i}", 0, 40, 6, tenant="big") for i in range(3)]
    requests += [SimRequest(f"small-{i}", 0, 8, 2, tenant="small") for i in range(10)]
    run = solve(ctx, requests, max_running=1)
    check.equal(
        [i for _, i in run.admissions],
        ["big-0", "small-0", "small-1", "small-2", "small-3", "small-4", "big-1", "small-5",
         "small-6", "small-7", "small-8", "big-2", "small-9"],
        label="admission order",
    )  # fmt: skip
    check_fairness(ctx, run)


@case("an idle tenant cannot bank credit: its counter is lifted when it returns",
      public=True)
def no_banked_credit(ctx):
    # Rejects no_lift. "busy" is served alone until t=190 while "idle" does nothing.
    # Then both are backlogged. With its old counter of 0, "idle" would be owed 116
    # and take every slot until its six requests are done.
    requests = [req(f"busy-{i}", 0, "busy") for i in range(16)]
    requests += [req(f"idle-{i}", 180, "idle") for i in range(6)]
    run = solve(ctx, requests, max_running=1, snapshot_at=(180.5,))
    expect_counters(run.snapshots[180.5], {"busy": 116, "idle": 116}, "counters at t=180.5", run)
    after = [i.split("-")[0] for when, i in run.admissions if when >= 180][:8]
    check.equal(after, ["idle", "busy", "idle", "busy", "idle", "busy", "idle", "busy"],
                label="who is admitted after t=180")  # fmt: skip
    check_fairness(ctx, run)


@case("a lift never lowers a counter")
def lift_never_lowers(ctx):
    # Rejects lift_without_max. "light" arrives at an empty queue and is lifted to
    # "heavy" (96). When the next request of "heavy" arrives at t=40, "heavy" stands at
    # 98 and has nothing queued; the queued tenant "light" is at 96. 98 stays 98.
    requests = [SimRequest("h0", 0, 96, 4, tenant="heavy"), req("l0", 1, "light"),
                req("l1", 2, "light"), req("l2", 3, "light"), req("h1", 40, "heavy")]
    run = solve(ctx, requests, max_running=1, snapshot_at=(40.5,))
    expect_counters(run.snapshots[40.5], {"heavy": 98, "light": 96}, "counters at t=40.5", run)
    check.equal([i for _, i in run.admissions], ["h0", "l0", "h1", "l1", "l2"],
                label="admission order")  # fmt: skip
    expect_counters(run.final_counters(), {"heavy": 116, "light": 132}, "final counters", run)


@case("with an empty queue, the lift is to the tenant that was admitted last")
def lift_with_an_empty_queue(ctx):
    # Rejects lift_skipped_when_queue_empty. At t=30 nothing is waiting; "b" was
    # admitted last and stands at 48 (40 input tokens and 4 output tokens so far), so
    # the new tenant "c" starts at 48. At t=31 "a" (12) finds "c" queued and is lifted too.
    requests = [req("a0", 0, "a"), SimRequest("b0", 1, 40, 6, tenant="b"), req("c0", 30, "c"),
                req("a1", 31, "a"), req("c1", 32, "c")]
    run = solve(ctx, requests, max_running=1, snapshot_at=(30.5, 31.5))
    expect_counters(run.snapshots[30.5], {"a": 12, "b": 48, "c": 48}, "counters at t=30.5", run)
    expect_counters(run.snapshots[31.5], {"a": 48, "b": 48, "c": 48}, "counters at t=31.5", run)
    check.equal([i for _, i in run.admissions], ["a0", "b0", "c0", "a1", "c1"],
                label="admission order")  # fmt: skip
    # The very first request of all: there is nobody to be lifted to.
    run = solve(ctx, [req("x", 7, "solo")], max_running=1, snapshot_at=(7.5,))
    expect_counters(run.snapshots[7.5], {"solo": 8}, "counters at t=7.5", run)


@case("equal counters: the tenant whose request is first in the queue; several slots at once")
def ties_and_several_slots(ctx):
    # Three tenants arrive together at counter 0. Each admission charges the prompt at
    # once, so one step's admissions spread over the tenants. At t=25 all three are at
    # 12: "z1" is first in the queue, then "a1" (queued before "m1").
    requests = [req("z0", 0, "z"), req("z1", 0, "z"), req("m0", 0, "m"), req("a0", 0, "a"),
                req("a1", 0, "a"), req("m1", 0, "m")]
    run = solve(ctx, requests, max_running=3)
    expect_admissions(run, [(0, "z0"), (0, "m0"), (0, "a0"), (25, "z1"), (25, "a1"), (25, "m1")])
    run = solve(ctx, requests, max_running=4)
    expect_admissions(run, [(0, "z0"), (0, "m0"), (0, "a0"), (0, "z1"), (28, "a1"), (28, "m1")])


@case("random traffic from four tenants: admissions, steps, and counters")
def random_workloads(ctx):
    for max_running, rate, weights in ((1, 0.012, (1.0, 2.0)), (3, 0.04, (1.0, 2.0)),
                                       (6, 0.07, (1.0, 4.0)), (4, 0.05, (2.0, 1.0))):  # fmt: skip
        requests = workloads.poisson(ctx.rng, rate=rate, n=100, prompt_tokens=(4, 96),
                                     output_tokens=(1, 24), tenants=("ann", "bo", "cy", "di"),
                                     quantum_s=1)  # fmt: skip
        run = matches_oracle(ctx, requests, max_running=max_running, w_in=weights[0],
                             w_out=weights[1])  # fmt: skip
        check_fairness(ctx, run, name=f"service_difference_{max_running}_slots")


@case("bursts: tenants go idle between bursts and return together")
def bursty_workloads(ctx):
    for max_running in (2, 4):
        requests = workloads.bursty(ctx.rng, bursts=6, burst_size=10, period_s=400,
                                    prompt_tokens=(4, 64), output_tokens=(1, 12),
                                    tenants=("ann", "bo", "cy"), quantum_s=1)  # fmt: skip
        run = matches_oracle(ctx, requests, max_running=max_running)
        check_fairness(ctx, run, name=f"service_difference_{max_running}_slots")


@case("three tenants backlogged for a long time: the service difference stays bounded")
def long_backlog_is_fair(ctx):
    # Rejects counts_requests on fairness itself: the tenants send very different
    # requests, all of them at t=0, and equal turns drift apart without bound.
    rng = ctx.rng
    requests, n = [], 0
    for tenant, prompts, outputs, count in (("chat", (4, 16), (2, 8), 150),
                                            ("docs", (48, 64), (1, 4), 60),
                                            ("essay", (4, 16), (12, 16), 60)):  # fmt: skip
        for _ in range(count):
            requests.append(SimRequest(n, 0, int(rng.integers(prompts[0], prompts[1] + 1)),
                                       int(rng.integers(outputs[0], outputs[1] + 1)),
                                       tenant=tenant))  # fmt: skip
            n += 1
    order = [int(i) for i in rng.permutation(len(requests))]
    requests = [requests[i] for i in order]
    run = solve(ctx, requests, max_running=4)
    check_discipline(run)
    check_fairness(ctx, run)
