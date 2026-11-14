"""Tests for priority-scheduler.

`simulate` runs the solution's `run` against a `SimEngine` on a simulated clock whose
unit is the millisecond. The statement determines which request gets every free slot,
so the cases check the admissions (time and id) and the steps exactly: against literal
schedules on hand-checked scenarios, and on seeded workloads against the selection
rule recomputed here from the statement and against the oracle.
"""

import asyncio
import dataclasses

from whetstone.judge import case, check
from whetstone.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, max_running, aging_ms):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.max_running = max_running
        self.aging_ms = aging_ms

    @property
    def steps(self):
        """[(start time, prefill, decode)] with prefill a tuple of (id, n) and decode of ids."""
        return [(e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")]

    @property
    def admissions(self):
        """[(time, id)] in admission order."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("admit")]

    def timeline(self):
        """A readable log of the simulation, for failure details."""
        settings = f"max_running={self.max_running}, aging_ms={self.aging_ms}"
        lines = [f"{settings}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        shown = [e for e in self.sim.trace.events if e.kind in ("arrive", "admit", "finish")]
        for e in shown[:80]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive  {r.id!r} (priority {r.priority}, prompt {r.prompt_tokens})"
            else:
                what = f"{e.kind:<7} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 80:
            lines.append(f"  ... {len(shown) - 80} more events")
        return "\n".join(lines) + "\n"


def horizon(requests):
    """A time by which any scheduler that keeps working has served everything."""
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    return requests[-1].arrival_s + 2 * serial + 1000


def simulate(ctx, scheduler, requests, *, max_running, aging_ms):
    """Deliver `requests` to `scheduler(engine, queue, max_running, aging_ms)`."""
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    queue = asyncio.Queue()
    workloads.drive(sim, requests, queue.put_nowait)
    sim.spawn(scheduler(engine, queue, max_running, aging_ms))
    run = Run(sim, engine, requests, max_running, aging_ms)
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
    return run


def solve(ctx, requests, *, max_running, aging_ms):
    return simulate(ctx, ctx.solution.run, requests, max_running=max_running, aging_ms=aging_ms)


def expect_admissions(run, expected):
    """Compare (time, id) admissions with a schedule, without leaking it in the message."""
    got = run.admissions
    detail = f"expected admissions (ms, id): {expected[:60]!r}\n"
    detail += f"got admissions      (ms, id): {got[:60]!r}\n" + run.timeline()
    for k, (have, want) in enumerate(zip(got, expected, strict=False), start=1):
        check.that(have[1] == want[1], f"admission {k} is the wrong request", detail)
        check.that(have[0] == want[0], f"admission {k} happens at the wrong time", detail)
    check.that(len(got) == len(expected), "the number of admissions is wrong", detail)


def selection_key(request, now_ms, aging_ms):
    """The statement's rule: effective priority, then arrival (then queue order)."""
    waited = now_ms - request.arrival_s
    return (request.priority - int(waited // aging_ms), request.arrival_s)


def check_selection(run):
    """Every admission took the request the statement's rule selects (no oracle)."""
    pending = list(run.requests)  # in arrival (= queue) order
    for k, (t, seq_id) in enumerate(run.admissions, start=1):
        waiting = [r for r in pending if r.arrival_s <= t]
        best = min(waiting, key=lambda r: selection_key(r, t, run.aging_ms), default=None)
        if best is None or best.id != seq_id:
            table = "".join(
                f"  {r.id!r}: priority {r.priority}, arrived {r.arrival_s:g}, "
                f"effective {selection_key(r, t, run.aging_ms)[0]}\n"
                for r in waiting[:12]
            )
            check.that(
                False,
                "a free slot did not go to the waiting request with the best "
                "(effective priority, arrival time)",
                f"admission {k} at t={t:g} took {seq_id!r}; waiting then:\n{table}"
                + run.timeline(),
            )
        pending.remove(best)


def matches_oracle(ctx, requests, *, max_running, aging_ms):
    got = solve(ctx, requests, max_running=max_running, aging_ms=aging_ms)
    check_selection(got)
    want = simulate(ctx, ctx.oracle.run, requests, max_running=max_running, aging_ms=aging_ms)
    expect_admissions(got, want.admissions)
    check.that(
        got.steps == want.steps,
        "the admissions are right but the engine steps differ from the continuous-batching loop",
        got.timeline(),
    )
    return got


def shuffled_ids(rng, requests):
    """The same workload with ids that are not in arrival order."""
    ids = [int(i) for i in rng.permutation(len(requests))]
    return [dataclasses.replace(r, id=i) for r, i in zip(requests, ids, strict=True)]


def req(id, arrival, priority, prompt=8, output=2):
    return SimRequest(id, arrival, prompt, output, priority=priority)


# The worked example. With one slot, a request with an 8-token prompt and 2 output
# tokens holds the slot for 10 + 9 = 19 ms.
EXAMPLE = [
    SimRequest(0, 0, 8, 3, priority=1),
    req(1, 2, 2),
    req(2, 4, 0),
    req(3, 6, 1),
    req(4, 6, 1),
    req(5, 60, 0),
]


@case("effective_priority: one level better per full aging_ms waited", public=True)
def effective_priority_table(ctx):
    fn = ctx.solution.effective_priority
    request = req("r", 100, 2)
    table = [(100, 2), (139.75, 2), (140, 1), (179, 1), (180, 0), (219.75, 0), (220, -1), (500, -8)]
    got = [fn(request, now, 40) for now, _ in table]
    check.that(
        all(type(v) is int for v in got),
        "effective_priority must return an int",
        f"returned: {got!r}\n",
    )
    check.equal(got, [want for _, want in table], label="effective priority over time")
    check.equal(fn(req("s", 0, 0), 9, 2.5), -3, label="effective priority with aging_ms=2.5")


@case("the worked example: priorities, a tie in queue order, and aging", public=True)
def worked_example(ctx):
    run = solve(ctx, EXAMPLE, max_running=1, aging_ms=40)
    expect_admissions(run, [(0, 0), (28, 2), (47, 3), (66, 4), (85, 1), (104, 5)])


@case("pick_next removes and returns the selected request", public=True)
def pick_next_directly(ctx):
    pick = ctx.solution.pick_next
    # The waiting list of the worked example at t=85.
    waiting = [EXAMPLE[1], EXAMPLE[5]]
    got = pick(waiting, 85, 40)
    check.that(got is EXAMPLE[1], "pick_next returned the wrong request at t=85",
               "waiting: request 1 (priority 2, arrived 2) and request 5 (priority 0, arrived 60)\n")
    check.that(waiting == [EXAMPLE[5]], "pick_next must remove the chosen request from `waiting`")
    # At t=47: request 1 has effective priority 1, requests 3 and 4 have 0 and tie on arrival.
    waiting = [EXAMPLE[1], EXAMPLE[3], EXAMPLE[4]]
    order = [pick(waiting, 47, 40).id for _ in range(3)]
    check.equal(order, [3, 4, 1], label="three picks at t=47")


@case("without aging in play: lower value first, first come first served within a level",
      public=True)
def strict_priorities(ctx):
    # Rejects larger_value_first. aging_ms is far longer than the scenario.
    requests = [req(0, 0, 5), req(1, 1, 2), req(2, 1, 0), req(3, 2, 1), req(4, 3, 0),
                req(5, 3, 2), req(6, 4, -1), req(7, 4, 1)]
    run = solve(ctx, requests, max_running=1, aging_ms=1_000_000)
    check.equal([i for _, i in run.admissions], [0, 6, 2, 4, 3, 7, 1, 5], label="admission order")
    # Two slots: request 0 holds one throughout, the other is handed out in the same order.
    run = solve(ctx, requests, max_running=2, aging_ms=1_000_000)
    expect_admissions(run, [(0, 0), (10, 6), (21, 2), (32, 4), (43, 3), (54, 7), (65, 1), (76, 5)])


@case("equal effective priority: the earlier arrival wins, whatever the ids", public=True)
def ties_go_to_the_earlier_arrival(ctx):
    # Rejects ties_by_request_id: ids are not in arrival order.
    requests = [req(9, 0, 0), req(7, 1, 1), req(3, 2, 1), req(8, 3, 1), req(1, 3, 1), req(5, 3, 1)]
    run = solve(ctx, requests, max_running=1, aging_ms=1_000_000)
    check.equal([i for _, i in run.admissions], [9, 7, 3, 8, 1, 5], label="admission order")
    # Aging makes the levels meet: at t=19 request "old" (priority 1, waited 18) and
    # request "new" (priority 0, waited 4) are both at level 0; the earlier arrival wins.
    requests = [req("first", 0, 0), req("old", 1, 1), req("new", 15, 0)]
    run = solve(ctx, requests, max_running=1, aging_ms=18)
    expect_admissions(run, [(0, "first"), (19, "old"), (38, "new")])


@case("a request that has waited exactly aging_ms has gained one level")
def aging_boundary(ctx):
    # At t=19, "low" (priority 1) has waited exactly 19 - 0.25 = 18.75 ms.
    requests = [req("first", 0, 0), req("low", 0.25, 1), req("high", 10, 0)]
    run = solve(ctx, requests, max_running=1, aging_ms=18.75)
    check.equal([i for _, i in run.admissions], ["first", "low", "high"], label="admission order")
    # A quarter of a millisecond short of a level: the fresh priority-0 request goes first.
    run = solve(ctx, requests, max_running=1, aging_ms=19)
    check.equal([i for _, i in run.admissions], ["first", "high", "low"], label="admission order")


@case("levels are whole: a partly aged request does not pass a better priority")
def aging_is_per_level(ctx):
    # Rejects continuous_aging. At t=19, "low" (priority 1, arrived 1) has waited 18 of
    # 20 ms: still level 1, behind "high" (priority 0, arrived 18).
    requests = [req("first", 0, 0), req("low", 1, 1), req("high", 18, 0)]
    run = solve(ctx, requests, max_running=1, aging_ms=20)
    check.equal([i for _, i in run.admissions], ["first", "high", "low"], label="admission order")
    # And levels keep falling below zero: at t=38 "low" (priority 3) has waited 37 ms
    # at 9 ms per level, level -1, ahead of "high" (priority 0, waited 8 ms).
    requests = [req("a", 0, 0), req("low", 1, 3), req("b", 2, 0), req("high", 30, 0)]
    run = solve(ctx, requests, max_running=1, aging_ms=9)
    check.equal([i for _, i in run.admissions], ["a", "b", "low", "high"], label="admission order")


@case("waiting time counts from the arrival, also for a request that arrived mid-step")
def wait_counts_from_arrival(ctx):
    # Rejects wait_from_collection. "low" arrives at 1, during the first step, and is
    # collected at 10. At t=38 it has waited 37 ms (one level at 30 ms per level), not 28.
    requests = [SimRequest("first", 0, 8, 2), req("mid", 2, 0), req("low", 1, 1),
                req("high", 30, 0)]
    run = solve(ctx, workloads.replay(requests), max_running=1, aging_ms=30)
    expect_admissions(run, [(0, "first"), (19, "mid"), (38, "low"), (57, "high")])


@case("priorities are evaluated when a slot is handed out, not when requests arrive")
def evaluated_at_admission(ctx):
    # Rejects aged_only_on_arrival. Nothing arrives after t=3. At t=28 "b" (priority 1)
    # is ahead. At t=47 "low" (priority 2, arrived 1) has waited 46 ms, two levels at
    # 23 ms per level, and "c" (priority 1, arrived 3) has waited 44 ms, one level:
    # both are at level 0 and the earlier arrival wins. Levels computed when the
    # requests arrived (2 and 1) would pick "c".
    requests = [SimRequest("first", 0, 8, 3), req("low", 1, 2), req("b", 2, 1), req("c", 3, 1)]
    run = solve(ctx, requests, max_running=1, aging_ms=23)
    expect_admissions(run, [(0, "first"), (28, "b"), (47, "low"), (66, "c")])


@case("random traffic with four priority levels and several aging rates")
def random_workloads(ctx):
    for max_running, rate, aging_ms in ((1, 0.01, 60), (3, 0.03, 25), (6, 0.05, 150), (4, 0.04, 7)):
        requests = workloads.poisson(ctx.rng, rate=rate, n=100, prompt_tokens=(4, 96),
                                     output_tokens=(1, 24), priorities=(0, 1, 2, 3), quantum_s=1)
        requests = shuffled_ids(ctx.rng, requests)
        matches_oracle(ctx, requests, max_running=max_running, aging_ms=aging_ms)


@case("bursts: many requests of mixed priority arrive at one instant")
def bursty_workloads(ctx):
    for max_running, aging_ms in ((2, 40), (5, 16)):
        requests = workloads.bursty(ctx.rng, bursts=5, burst_size=9, period_s=96,
                                    prompt_tokens=(4, 64), output_tokens=(1, 10),
                                    priorities=(0, 0, 1, 2), quantum_s=1)
        matches_oracle(ctx, shuffled_ids(ctx.rng, requests), max_running=max_running,
                       aging_ms=aging_ms)


@case("a steady high-priority stream does not starve low-priority requests")
def no_starvation(ctx):
    # Rejects no_aging. One request in eight has priority 3; the rest are priority 0 and
    # arrive faster than four slots can serve them, so a priority-0 request is always
    # waiting. Without aging the priority-3 requests wait for the stream to end.
    requests = workloads.poisson(ctx.rng, rate=0.021, n=320, prompt_tokens=(16, 64),
                                 output_tokens=(4, 24), priorities=(0, 0, 0, 0, 0, 0, 0, 3),
                                 quantum_s=1)
    run = solve(ctx, requests, max_running=4, aging_ms=200)
    admitted = {i: t for t, i in run.admissions}
    low = [admitted[r.id] - r.arrival_s for r in requests if r.priority == 3]
    high = [admitted[r.id] - r.arrival_s for r in requests if r.priority == 0]
    ctx.metric("longest_high_priority_wait", max(high), unit="ms")
    check.at_most("longest_low_priority_wait", max(low), 2000.0, unit="ms")
