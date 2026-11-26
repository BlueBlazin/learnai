"""Tests for continuous-batching-scheduler.

`simulate` runs a scheduler coroutine against a `SimEngine` on a simulated clock whose
unit is the millisecond. The policy in the statement determines every step, so each
case checks the list of steps (start time, prefill, decode): against literal schedules
on the hand-checked scenarios, and on seeded workloads against invariants computed
here from the statement and against the oracle's steps.
"""

import asyncio
from collections import deque

from learnai.judge import case, check, percentile
from learnai.kit import CostModel, ProtocolViolation, SimEngine, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, max_running):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.max_running = max_running

    @property
    def steps(self):
        """[(start time, prefill, decode)] with prefill a tuple of (id, n) and decode of ids."""
        return [(e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")]

    def timeline(self):
        """A readable log of the simulation, for failure details."""
        lines = [f"max_running={self.max_running}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        shown = [e for e in self.sim.trace.events if e.kind in ("arrive", "step", "finish")]
        for e in shown[:70]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive  {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            else:
                what = f"finish  {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 70:
            lines.append(f"  ... {len(shown) - 70} more events")
        return "\n".join(lines) + "\n"


def horizon(requests):
    """A time by which any scheduler that keeps working has served everything."""
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    return requests[-1].arrival_s + 2 * serial + 1000


def simulate(ctx, scheduler, requests, *, max_running):
    """Deliver `requests` to `scheduler(engine, queue, max_running)` and run to the end."""
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    queue = asyncio.Queue()
    workloads.drive(sim, requests, queue.put_nowait)
    sim.spawn(scheduler(engine, queue, max_running))
    run = Run(sim, engine, requests, max_running)
    try:
        # A bounded run: a scheduler that polls on a timer never lets the clock go idle.
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
    return run


def check_served(run):
    """Every request finished, with all of its tokens."""
    finished = {e["id"]: e["tokens"] for e in run.sim.trace.of("finish")}
    missing = [r.id for r in run.requests if r.id not in finished]
    check.that(
        not missing,
        f"{len(missing)} of {len(run.requests)} requests never finished",
        f"unfinished: {missing[:10]!r}\n" + run.timeline(),
    )
    check.that(
        not run.engine.live and not run.engine.busy,
        "the engine still holds sequences at the end of the simulation",
        run.timeline(),
    )


def expect_steps(run, expected):
    """Compare a run's steps with a schedule."""
    got = run.steps
    detail = "expected steps (start ms, prefill, decode):\n"
    detail += "".join(f"  {step!r}\n" for step in expected[:40])
    detail += "got:\n" + "".join(f"  {step!r}\n" for step in got[:40]) + run.timeline()
    for k, (have, want) in enumerate(zip(got, expected, strict=False)):
        check.that(
            have[1] == want[1], f"step {k} prefills {have[1]!r}, expected {want[1]!r}", detail
        )
        check.that(
            have[2] == want[2], f"step {k} decodes {have[2]!r}, expected {want[2]!r}", detail
        )
        check.that(
            have[0] == want[0], f"step {k} starts at {have[0]!r} ms, expected {want[0]!r}", detail
        )
    check.that(
        len(got) == len(expected),
        f"the scheduler ran {len(got)} steps, expected {len(expected)}",
        detail,
    )


def check_invariants(run):
    """The policy, restated as properties of the trace (independent of the oracle)."""
    check_served(run)
    order = deque(run.requests)  # not admitted yet, in arrival order
    live = []  # admitted and unfinished, in admission order
    finished_in = {}
    for e in run.sim.trace.of("finish"):
        finished_in.setdefault(e["step"], set()).add(e["id"])
    free_at = 0.0  # when the previous step ended
    for e in run.sim.trace.of("step"):
        t, k = e.time, e["index"]

        def need(condition, message, extra="", e=e):
            if not condition:  # the detail is built only for a failure
                where = f"step {e['index']} at t={e.time:g}: prefill={list(e['prefill'])} "
                where += f"decode={list(e['decode'])}\n"
                check.that(False, message, where + extra + run.timeline())

        # When the engine could first have started this step.
        if live or (order and order[0].arrival_s <= free_at):
            earliest = free_at
        else:
            earliest = order[0].arrival_s if order else t
        need(t == earliest, "the engine sat idle although there was work to do")
        need(
            tuple(e["decode"]) == tuple(live),
            "a step must decode every running sequence, in admission order",
            f"running before this step: {live!r}\n",
        )
        for seq_id, n in e["prefill"]:
            head = order.popleft() if order else None
            need(
                head is not None and head.id == seq_id,
                "requests were not admitted first come, first served",
            )
            need(n == head.prompt_tokens, "a prompt was not prefilled whole in one step")
            live.append(seq_id)
        need(len(live) <= run.max_running, "more than max_running sequences are live")
        need(
            not (order and order[0].arrival_s <= t) or len(live) == run.max_running,
            "a slot was free at the start of a step while a request was waiting",
            f"live in this step: {live!r}\n",
        )
        live = [i for i in live if i not in finished_in.get(k, ())]
        free_at = t + e["duration"]


def solve(ctx, requests, *, max_running):
    return simulate(ctx, ctx.solution.run, requests, max_running=max_running)


def matches_oracle(ctx, requests, *, max_running):
    got = solve(ctx, requests, max_running=max_running)
    check_invariants(got)
    want = simulate(ctx, ctx.oracle.run, requests, max_running=max_running)
    expect_steps(got, want.steps)
    return got


async def static_batching(engine, queue, max_running):
    """The baseline the statement compares with: a new batch starts only when the
    previous one has finished entirely."""
    waiting, running = deque(), []
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())
        prefill = []
        while not running and waiting and len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
        result = await engine.step(prefill=prefill, decode=running)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in result.finished]


@case("the worked example: joining mid-flight, leaving early, and reusing a slot", public=True)
def worked_example(ctx):
    rows = [(0, 8, 4), (0, 16, 2), (5, 8, 3), (45, 4, 2), (50, 12, 1), (60, 4, 2)]
    run = solve(ctx, workloads.replay(rows), max_running=2)
    check_served(run)
    expect_steps(run, [
        (0, ((0, 8), (1, 16)), ()),
        (14, (), (0, 1)),
        (24, ((2, 8),), (0,)),
        (35, (), (0, 2)),
        (45, ((3, 4),), (2,)),
        (55, ((4, 12),), (3,)),
        (67, ((5, 4),), ()),
        (76, (), (5,)),
    ])  # fmt: skip
    check.equal(run.sim.trace.latencies("arrive", "finish"), [45.0, 24.0, 50.0, 22.0, 17.0, 25.0],
                label="end-to-end latencies (ms)")  # fmt: skip


@case("a lone request: one prefill step, then one decode step per remaining token", public=True)
def lone_request(ctx):
    run = solve(ctx, workloads.replay([(3, 40, 4)]), max_running=4)
    check_served(run)
    expect_steps(run, [(3, ((0, 40),), ()), (21, (), (0,)), (30, (), (0,)), (39, (), (0,))])
    # A single output token comes out of the prefill step; no decode step follows.
    run = solve(ctx, workloads.replay([(0, 8, 1), (100, 8, 1)]), max_running=4)
    check_served(run)
    expect_steps(run, [(0, ((0, 8),), ()), (100, ((1, 8),), ())])


@case("a request that arrives during a step joins the next step", public=True)
def joins_mid_flight(ctx):
    # Rejects waits_for_whole_batch: request 1 must not wait for request 0 to finish.
    run = solve(ctx, workloads.replay([(0, 16, 6), (20, 8, 2)]), max_running=4)
    check_served(run)
    expect_steps(run, [
        (0, ((0, 16),), ()),
        (12, (), (0,)),
        (21, ((1, 8),), (0,)),
        (32, (), (0, 1)),
        (42, (), (0,)),
        (51, (), (0,)),
    ])  # fmt: skip


@case("a burst larger than the slots: first come, first served, and no slot idles", public=True)
def burst_over_capacity(ctx):
    # Rejects newest_first, admits_one_per_step, and drops_requests_without_slot.
    rows = [(0, 4, 2), (0, 4, 3), (0, 4, 2), (0, 4, 1), (0, 4, 2)]
    run = solve(ctx, workloads.replay(rows), max_running=3)
    check_served(run)
    expect_steps(run, [
        (0, ((0, 4), (1, 4), (2, 4)), ()),
        (11, (), (0, 1, 2)),
        (22, ((3, 4), (4, 4)), (1,)),
        (33, (), (4,)),
    ])  # fmt: skip


@case("a finished sequence frees its slot for the very next step", public=True)
def slot_reused_at_once(ctx):
    # Rejects slot_idles_one_step: request 2 takes the slot of request 0 at t=20.
    run = solve(ctx, workloads.replay([(0, 4, 2), (0, 4, 5), (1, 4, 2)]), max_running=2)
    check_served(run)
    expect_steps(run, [
        (0, ((0, 4), (1, 4)), ()),
        (10, (), (0, 1)),
        (20, ((2, 4),), (1,)),
        (30, (), (1, 2)),
        (40, (), (1,)),
    ])  # fmt: skip


@case("arrivals at the exact instant a step ends are admitted into the next step")
def arrival_when_a_step_ends(ctx):
    # Step 0 runs from 0 to 10. Requests 1 and 2 arrive at exactly 10, request 3 at 10.5.
    rows = [(0, 8, 3), (10, 4, 1), (10, 8, 2), (10.5, 4, 1)]
    run = solve(ctx, workloads.replay(rows), max_running=3)
    check_served(run)
    expect_steps(run, [
        (0, ((0, 8),), ()),
        (10, ((1, 4), (2, 8)), (0,)),
        (22, ((3, 4),), (0, 2)),
    ])  # fmt: skip


@case("the engine goes idle and wakes at the next arrival; request ids are opaque")
def idle_gaps_and_opaque_ids(ctx):
    rows = [
        {"id": "b", "arrival_s": 5, "prompt_tokens": 8, "output_tokens": 2},
        {"id": "a", "arrival_s": 100, "prompt_tokens": 4, "output_tokens": 1},
        {"id": "z", "arrival_s": 100, "prompt_tokens": 4, "output_tokens": 2},
        {"id": "c", "arrival_s": 100, "prompt_tokens": 4, "output_tokens": 2},
        {"id": "m", "arrival_s": 300.25, "prompt_tokens": 12, "output_tokens": 1},
    ]
    run = solve(ctx, workloads.replay(rows), max_running=2)
    check_served(run)
    expect_steps(run, [
        (5, (("b", 8),), ()),
        (15, (), ("b",)),
        (100, (("a", 4), ("z", 4)), ()),
        (110, (("c", 4),), ("z",)),
        (120, (), ("c",)),
        (300.25, (("m", 12),), ()),
    ])  # fmt: skip


@case("max_running = 1 is one request at a time, in arrival order")
def one_slot(ctx):
    rows = [(0, 8, 2), (1, 4, 1), (2, 4, 3), (200, 4, 1)]
    run = matches_oracle(ctx, workloads.replay(rows), max_running=1)
    check.equal([p[0][0] for _, p, _ in run.steps if p], [0, 1, 2, 3], label="admission order")


@case("bursts that arrive at one instant, some exactly when a step ends")
def bursty_workloads(ctx):
    for max_running in (2, 5, 8):
        requests = workloads.bursty(ctx.rng, bursts=6, burst_size=7, period_s=64,
                                    prompt_tokens=(4, 64), output_tokens=(1, 12), quantum_s=1)
        matches_oracle(ctx, requests, max_running=max_running)


@case("random traffic: every step is the one the policy dictates")
def random_workloads(ctx):
    for max_running, rate in ((1, 0.004), (3, 0.02), (8, 0.05), (16, 0.05)):
        # Arrivals are whole milliseconds, so some coincide with each other.
        requests = workloads.poisson(ctx.rng, rate=rate, n=120, prompt_tokens=(4, 128),
                                     output_tokens=(1, 40), quantum_s=1)
        matches_oracle(ctx, requests, max_running=max_running)


@case("under load, median latency is less than half of static batching's")
def beats_static_batching(ctx):
    requests = workloads.poisson(ctx.rng, rate=0.012, n=150, prompt_tokens=(16, 128),
                                 output_tokens=(2, 40), quantum_s=1)
    run = solve(ctx, requests, max_running=8)
    check_served(run)
    static = simulate(ctx, static_batching, requests, max_running=8)
    static_p50 = percentile(static.sim.trace.latencies("arrive", "finish"), 50)
    static_ttft = percentile(static.sim.trace.latencies("arrive", "first_token"), 99)
    ctx.metric("static_p50_latency", static_p50, unit="ms")
    check.at_most("p50_latency", percentile(run.sim.trace.latencies("arrive", "finish"), 50),
                  0.5 * static_p50, unit="ms")  # fmt: skip
    check.at_most("p99_ttft", percentile(run.sim.trace.latencies("arrive", "first_token"), 99),
                  0.5 * static_ttft, unit="ms")  # fmt: skip
