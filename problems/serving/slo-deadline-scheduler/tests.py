"""Tests for slo-deadline-scheduler.

`plan_step` is called directly on small waiting lists with literal expectations.
`run` is simulated on a `SimEngine` whose clock counts milliseconds: the test records
every engine step and every `on_shed` call, and checks them against literal values,
against rules computed here from the statement, and against the oracle. The overload
case computes SLO attainment as `goodput-slo` defines it: requests whose first token
came by their deadline, divided by all requests, shed ones included.
"""

import asyncio
import dataclasses

from learnai.judge import case, check
from learnai.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


def req(id, arrival, prompt, output, deadline=None):
    return SimRequest(id, arrival, prompt, output, deadline_s=deadline)


def w(id, prompt, deadline):
    """A waiting request for direct `plan_step` calls."""
    return SimRequest(id, 0.0, prompt, 1, deadline_s=deadline)


def show_waiting(waiting):
    return "[" + ", ".join(
        f"{r.id!r}(prompt {r.prompt_tokens}, deadline {r.deadline_s})" for r in waiting
    ) + "]"  # fmt: skip


def check_plan(ctx, waiting, now, running, free, *, admit, shed, cost=COST, label="plan_step"):
    """Call the solution's `plan_step` on a copy of `waiting` and compare ids."""
    given = list(waiting)
    detail = (
        f"plan_step(waiting, now_ms={now!r}, running={running}, free_slots={free}, step_ms)\n"
        f"  waiting = {show_waiting(waiting)}\n"
        f"  step_ms(P, D) = {cost.base_s:g} + {cost.prefill_s_per_token:g} * P + "
        f"{cost.decode_s_per_seq:g} * D\n"
        f"expected: admit {admit!r}, shed {shed!r}\n"
    )
    result = ctx.solution.plan_step(given, now, running, free, cost.step_s)
    check.that(
        isinstance(result, tuple) and len(result) == 2,
        f"{label}: must return a tuple (admit, shed)",
        detail,
    )
    got_admit, got_shed = ([r.id for r in part] for part in result)
    detail += f"got: admit {got_admit!r}, shed {got_shed!r}\n"
    check.that(got_shed == shed, f"{label}: the wrong requests were shed", detail)
    check.that(got_admit == admit, f"{label}: the wrong requests were admitted, or in the "
               "wrong order", detail)  # fmt: skip
    left = [r.id for r in waiting if r.id not in admit and r.id not in shed]
    check.that(
        [r.id for r in given] == left,
        f"{label}: `waiting` must keep exactly the other requests, in their order",
        detail + f"waiting afterwards: {[r.id for r in given]!r}, expected {left!r}\n",
    )


class Run:
    def __init__(self, sim, engine, requests, max_running):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.max_running = max_running

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [
            (e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")
        ]

    @property
    def sheds(self):
        """[(time, id)] of `on_shed` calls."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("shed")]

    @property
    def first_tokens(self):
        return {e["id"]: e.time for e in self.sim.trace.of("first_token")}

    def attainment(self):
        """Fraction of all requests whose first token came by their deadline."""
        first = self.first_tokens
        good = sum(
            1
            for r in self.requests
            if r.id in first and (r.deadline_s is None or first[r.id] <= r.deadline_s)
        )
        return good / len(self.requests)

    def timeline(self):
        lines = [f"max_running={self.max_running}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        kinds = ("arrive", "step", "first_token", "finish", "shed")
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:90]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens}"
                what += f", deadline {r.deadline_s:g})" if r.deadline_s is not None else ")"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "shed":
                what = f"on_shed({e['id']!r})"
            else:
                what = f"{e.kind:<11} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 90:
            lines.append(f"  ... {len(shown) - 90} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, module, requests, *, max_running):
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    queue = asyncio.Queue()
    run = Run(sim, engine, requests, max_running)
    workloads.drive(sim, requests, queue.put_nowait)
    sim.spawn(module.run(engine, queue, max_running, lambda r: sim.trace.record("shed", id=r.id)))
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    try:
        sim.run(until=requests[-1].arrival_s + 2 * serial + 1000)
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


def check_rules(run):
    """What the statement promises, whatever the schedule."""
    detail = run.timeline()
    first = run.first_tokens
    finished = {e["id"] for e in run.sim.trace.of("finish")}
    admitted = {e["id"] for e in run.sim.trace.of("admit")}
    shed = [i for _, i in run.sheds]
    check.that(len(shed) == len(set(shed)), "a request was shed twice", detail)
    for r in run.requests:
        who = f"request {r.id!r}: "
        if r.id in shed:
            check.that(r.id not in admitted, "a request was shed and admitted", who + "\n" + detail)
            continue
        check.that(r.id in finished, "a request was neither shed nor served to the end",
                   who + "\n" + detail)  # fmt: skip
        check.that(
            r.deadline_s is None or first[r.id] <= r.deadline_s,
            "a request was served after its deadline instead of being shed",
            f"{who}first token at t={first[r.id]:g}, deadline {r.deadline_s}\n" + detail,
        )


def expect(run, *, steps=None, sheds=None):
    detail = run.timeline()
    if sheds is not None:
        check.that(
            run.sheds == sheds,
            "on_shed was not called for the right requests at the right times, in order",
            f"expected (time, id): {sheds!r}\ngot: {run.sheds!r}\n" + detail,
        )
    if steps is not None:
        got = run.steps
        lines = "expected steps (start ms, prefill, decode):\n"
        lines += "".join(f"  {step!r}\n" for step in steps[:40])
        lines += "got:\n" + "".join(f"  {step!r}\n" for step in got[:40]) + detail
        for k, (have, want) in enumerate(zip(got, steps, strict=False)):
            check.that(have[1] == want[1], f"step {k} prefills the wrong sequences", lines)
            check.that(have[2] == want[2], f"step {k} decodes the wrong sequences", lines)
            check.that(have[0] == want[0], f"step {k} starts at the wrong time", lines)
        check.that(len(got) == len(steps), "the number of engine steps is wrong", lines)


def solve(ctx, requests, *, max_running):
    run = simulate(ctx, ctx.solution, requests, max_running=max_running)
    check_rules(run)
    return run


def matches_oracle(ctx, requests, *, max_running):
    got = solve(ctx, requests, max_running=max_running)
    want = simulate(ctx, ctx.oracle, requests, max_running=max_running)
    expect(got, steps=want.steps, sheds=want.sheds)
    return got


def with_deadlines(rng, requests, budgets, *, none_share=0.0):
    """Give each request a TTFT budget drawn from `budgets` (ms after arrival)."""
    out = []
    for r in requests:
        budget = float(rng.choice(budgets))
        keep = rng.random() >= none_share
        out.append(dataclasses.replace(r, deadline_s=r.arrival_s + budget if keep else None))
    return out


def overload(rng):
    """The overload fixture: 300 requests at 40 per second for four slots (about
    twice what they can serve); half need a first token within 150 ms, half within
    1200 ms."""
    requests = workloads.poisson(rng, rate=0.04, n=300, prompt_tokens=(16, 160),
                                 output_tokens=(4, 24), quantum_s=0.25)  # fmt: skip
    return with_deadlines(rng, requests, [150.0, 1200.0])


def bursts(rng):
    """The burst fixture: 24 bursts of 12 requests, 700 ms apart, for four slots. On
    average the slots keep up; inside a burst they do not. Half of the requests need
    a first token within 200 ms, half within 1500 ms."""
    requests = workloads.bursty(rng, bursts=24, burst_size=12, period_s=700, jitter_s=40,
                                prompt_tokens=(16, 160), output_tokens=(4, 24),
                                quantum_s=0.25)  # fmt: skip
    return with_deadlines(rng, requests, [200.0, 1500.0])


def require_attainment(ctx, run, limit):
    value = round(run.attainment(), 4)
    ctx.metric("slo_attainment", value, None, limit, higher_is_better=True)
    check.that(
        value >= limit,
        f"slo_attainment: {value} is below the required {limit}",
        f"{len(run.sheds)} of {len(run.requests)} requests were shed\n",
    )


@case("the worked example: an urgent request overtakes, one is shed, one waits a step",
      public=True)  # fmt: skip
def worked_example(ctx):
    requests = [req("A", 0, 16, 3, 100), req("B", 2, 40, 2, 60), req("C", 4, 16, 2, 30),
                req("D", 6, 80, 2, 45), req("E", 8, 32, 2)]  # fmt: skip
    run = solve(ctx, requests, max_running=2)
    expect(
        run,
        steps=[(0, (("A", 16),), ()), (12, (("C", 16),), ("A",)), (25, (), ("A", "C")),
               (35, (("B", 40),), ()), (53, (("E", 32),), ("B",)), (70, (), ("E",))],
        sheds=[(25, "D")],
    )  # fmt: skip
    check.equal(run.first_tokens, {"A": 12, "C": 25, "B": 53, "E": 70}, label="first token times")


@case("plan_step: earliest deadline first, and a request that cannot be on time is shed",
      public=True)  # fmt: skip
def plan_orders_and_sheds(ctx):
    # Rejects fcfs_order: three slots, an idle engine, deadlines out of arrival order.
    waiting = [w("a", 16, 90), w("b", 16, 40), w("c", 16, 60)]
    check_plan(ctx, waiting, 0, 0, 3, admit=["b", "c", "a"], shed=[])
    # Rejects serves_hopeless_requests and sheds_only_expired: alone on an idle
    # engine "late" would get its first token at 100 + 8 + 20 = 128, after 120.
    waiting = [w("late", 80, 120), w("ok", 16, 150)]
    check_plan(ctx, waiting, 100, 0, 2, admit=["ok"], shed=["late"])
    # A deadline that has already passed.
    check_plan(ctx, [w("old", 4, 99)], 100, 0, 1, admit=[], shed=["old"])
    # No deadline: never shed, and behind every request that has one.
    waiting = [w("free", 16, None), w("due", 16, 500)]
    check_plan(ctx, waiting, 0, 0, 2, admit=["due", "free"], shed=[])


@case("plan_step: the estimate counts the prompts admitted ahead and the running decodes",
      public=True)  # fmt: skip
def plan_counts_the_whole_step(ctx):
    # Rejects ignores_requests_ahead: alone "y" would be done at 8 + 10 = 18, but
    # behind "x" the step takes 8 + 0.25 * 120 = 38 > 30.
    waiting = [w("x", 80, 28), w("y", 40, 30)]
    check_plan(ctx, waiting, 0, 0, 2, admit=["x"], shed=["y"])
    # Rejects ignores_decode_cost: 8 + 0.25 * 40 + 4 decodes = 22 > 20.
    check_plan(ctx, [w("z", 40, 20)], 0, 4, 1, admit=[], shed=["z"])
    check_plan(ctx, [w("z", 40, 20)], 0, 2, 1, admit=["z"], shed=[])


@case("plan_step: a first token at exactly the deadline is on time", public=True)
def plan_exact_deadline(ctx):
    # Rejects sheds_at_exact_deadline: 50 + 8 + 0.25 * 32 + 2 = 68.
    check_plan(ctx, [w("edge", 32, 68)], 50, 2, 1, admit=["edge"], shed=[])
    check_plan(ctx, [w("edge", 32, 67.75)], 50, 2, 1, admit=[], shed=["edge"])
    # The same boundary for a request behind another one in the step.
    waiting = [w("p", 16, 30), w("q", 16, 16)]
    check_plan(ctx, waiting, 0, 0, 2, admit=["q", "p"], shed=[])


@case("plan_step: a request that would make an admitted one late waits for a later step",
      public=True)  # fmt: skip
def plan_keeps_earlier_deadlines(ctx):
    # Rejects breaks_earlier_deadlines: with "big" in the step, "tight" would get
    # its first token at 8 + 0.25 * 176 = 52 > 20. "big" is not hopeless (52 <= 400).
    waiting = [w("tight", 16, 20), w("big", 160, 400)]
    check_plan(ctx, waiting, 0, 0, 2, admit=["tight"], shed=[])
    # Rejects stops_at_first_misfit: "small" behind "big" still fits (8 + 8 = 16 <= 20),
    # and "gone" behind both is hopeless and is shed now.
    waiting = [w("tight", 16, 20), w("big", 160, 400), w("small", 16, 450), w("gone", 400, 500)]
    check_plan(ctx, waiting, 0, 0, 3, admit=["tight", "small"], shed=[])
    waiting = [w("tight", 16, 20), w("big", 160, 400), w("gone", 2000, 500)]
    check_plan(ctx, waiting, 0, 0, 3, admit=["tight"], shed=["gone"])


@case("bursts: SLO attainment with deadlines of 200 ms and 1500 ms", public=True)
def burst_attainment(ctx):
    # Rejects fcfs_order (about 0.68) as well as every policy that serves late.
    run = solve(ctx, bursts(ctx.rng), max_running=4)
    require_attainment(ctx, run, 0.78)


@case("plan_step: without a free slot nothing is admitted, and hopeless requests are "
      "still shed")  # fmt: skip
def plan_without_free_slots(ctx):
    # Rejects sheds_only_while_slots_are_free.
    waiting = [w("a", 16, 100), w("b", 200, 60), w("c", 16, 15)]
    check_plan(ctx, waiting, 0, 4, 0, admit=[], shed=["c", "b"])
    waiting = [w("a", 16, 100), w("b", 16, 90), w("c", 300, 80), w("d", 16, 95)]
    check_plan(ctx, waiting, 0, 1, 1, admit=["b"], shed=["c"])
    check_plan(ctx, [], 0, 2, 2, admit=[], shed=[])


@case("plan_step: equal deadlines are served in the order of `waiting`")
def plan_ties(ctx):
    # Rejects ties_by_id: ids are opaque and not in arrival order.
    waiting = [w("zeta", 16, 50), w("alpha", 16, 50), w(7, 16, 50), w("mid", 16, 40)]
    check_plan(ctx, waiting, 0, 0, 2, admit=["mid", "zeta"], shed=[])
    waiting = [w("zeta", 16, None), w("alpha", 16, None)]
    check_plan(ctx, waiting, 0, 0, 1, admit=["zeta"], shed=[])
    # Shed requests are reported in the order they were considered.
    waiting = [w("zeta", 400, 50), w("alpha", 400, 50), w("beta", 400, 20)]
    check_plan(ctx, waiting, 0, 0, 3, admit=[], shed=["beta", "zeta", "alpha"])


@case("plan_step: another cost model, passed as step_ms")
def plan_other_cost_model(ctx):
    cost = CostModel(base_s=2, prefill_s_per_token=0.5, decode_s_per_seq=3)
    # 2 + 0.5 * 20 + 3 * 2 = 18.
    check_plan(ctx, [w("a", 20, 118)], 100, 2, 1, admit=["a"], shed=[], cost=cost)
    check_plan(ctx, [w("a", 20, 117.5)], 100, 2, 1, admit=[], shed=["a"], cost=cost)
    waiting = [w("a", 20, 130), w("b", 10, 124), w("c", 10, 125)]
    check_plan(ctx, waiting, 100, 2, 3, admit=["b", "c"], shed=[], cost=cost)


@case("everything waiting is shed: no empty step, and the loop serves the next arrival")
def all_shed(ctx):
    requests = [req("a", 0, 16, 2, 100), req("b", 1, 400, 2, 60), req("c", 2, 400, 2, 70),
                req("d", 200, 16, 1, 300)]  # fmt: skip
    run = solve(ctx, requests, max_running=1)
    expect(
        run,
        steps=[(0, (("a", 16),), ()), (12, (), ("a",)), (200, (("d", 16),), ())],
        sheds=[(12, "b"), (12, "c")],
    )
    # A lone request that is hopeless on arrival, on an idle engine.
    run = solve(ctx, [req("x", 5, 400, 1, 50), req("y", 5, 16, 1, 50)], max_running=2)
    expect(run, steps=[(5, (("y", 16),), ())], sheds=[(5, "x")])


@case("sustained overload: nobody is served late, and attainment stays near capacity")
def overload_attainment(ctx):
    # Rejects serves_hopeless_requests (about 0.03) and sheds_only_expired (about 0.2).
    run = solve(ctx, overload(ctx.rng), max_running=4)
    require_attainment(ctx, run, 0.4)


@case("random traffic with mixed deadlines: steps and sheds equal the reference's")
def random_workloads(ctx):
    for max_running, rate, budgets in (
        (1, 0.01, [40.0, 200.0]), (3, 0.03, [60.0, 150.0, 800.0]), (6, 0.06, [50.0, 400.0]),
        (4, 0.05, [30.0, 90.0, 2000.0]),
    ):  # fmt: skip
        requests = workloads.poisson(ctx.rng, rate=rate, n=80, prompt_tokens=(4, 160),
                                     output_tokens=(1, 24), quantum_s=1)  # fmt: skip
        requests = with_deadlines(ctx.rng, requests, budgets, none_share=0.15)
        matches_oracle(ctx, requests, max_running=max_running)


@case("bursts on whole milliseconds: equal deadlines and first tokens exactly at the deadline")
def bursty_workloads(ctx):
    for max_running in (2, 4):
        # 8-token prompts and whole-millisecond budgets: steps last whole milliseconds,
        # so many first tokens fall exactly on a deadline and many deadlines tie.
        requests = workloads.bursty(ctx.rng, bursts=8, burst_size=8, period_s=128,
                                    prompt_tokens=8, output_tokens=(1, 8), quantum_s=1)  # fmt: skip
        requests = with_deadlines(ctx.rng, requests, [11.0, 14.0, 23.0, 36.0, 60.0])
        ids = ctx.rng.permutation(len(requests))
        requests = [
            dataclasses.replace(r, id=f"r{int(ids[k]):02d}") for k, r in enumerate(requests)
        ]
        matches_oracle(ctx, requests, max_running=max_running)
