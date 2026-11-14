"""Tests for chunked-prefill-scheduling.

`plan` is a pure function, so half of the cases call it directly. The other half run
it inside `serve`, the loop from the statement, against a `SimEngine` with a token
budget on a simulated clock whose unit is the millisecond, and check the steps and the
latency numbers the statement promises.
"""

import asyncio
from collections import deque
from itertools import pairwise

from whetstone.judge import case, check
from whetstone.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


def checked_plan(plan, running, prefilling, waiting, budget, threshold):
    """Call `plan` on copies of the lists and insist on the return type."""
    got = plan(list(running), list(prefilling), list(waiting), budget, threshold)
    described = (
        f"plan(running={running!r}, prefilling={prefilling!r}, waiting={waiting!r}, "
        f"max_batch_tokens={budget}, long_prefill_threshold={threshold})\nreturned {got!r}\n"
    )
    check.that(
        isinstance(got, tuple) and len(got) == 2 and all(isinstance(x, list) for x in got),
        "plan must return a tuple (prefill, decode) of two lists",
        described,
    )
    prefill, decode = got
    check.that(
        all(isinstance(c, tuple) and len(c) == 2 and type(c[1]) is int for c in prefill),
        "every prefill entry must be a tuple (id, n_tokens) with an int n_tokens",
        described,
    )
    return prefill, decode


def expect_plan(ctx, running, prefilling, waiting, budget, threshold, want):
    """One direct call compared with a literal (prefill, decode)."""
    prefill, decode = checked_plan(ctx.solution.plan, running, prefilling, waiting, budget,
                                   threshold)  # fmt: skip
    detail = (
        f"plan(running={running!r}, prefilling={prefilling!r}, waiting={waiting!r}, "
        f"max_batch_tokens={budget}, long_prefill_threshold={threshold})\n"
        f"expected (prefill, decode): {want!r}\ngot: {(prefill, decode)!r}\n"
    )
    check.that(decode == want[1], "plan returned the wrong decode list", detail)
    check.that(prefill == want[0], "plan returned the wrong prefill chunks", detail)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, budget, threshold):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.budget = budget
        self.threshold = threshold

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [(e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")]

    def token_times(self):
        """{id: [time of each output token]}."""
        times = {}
        for e in self.sim.trace.of("token"):
            times.setdefault(e["id"], []).append(e.time)
        return times

    def timeline(self):
        settings = f"max_batch_tokens={self.budget}, long_prefill_threshold={self.threshold}"
        lines = [f"{settings}; a step takes 8 + 0.25 * P + D ms", "timeline:"]
        shown = [e for e in self.sim.trace.events if e.kind in ("arrive", "step", "finish")]
        for e in shown[:70]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive  {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['tokens']} tokens, {e['duration']:g} ms)"
            else:
                what = f"finish  {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 70:
            lines.append(f"  ... {len(shown) - 70} more events")
        return "\n".join(lines) + "\n"


async def serve(engine, queue, plan, max_batch_tokens, long_prefill_threshold):
    """The loop from the statement: it owns the state, `plan` decides each step."""
    waiting = deque()  # requests that are not admitted yet, in arrival order
    while True:
        if not waiting and not engine.live and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        left = {seq_id: engine.remaining_prefill(seq_id) for seq_id in engine.live}
        running = [seq_id for seq_id, n in left.items() if n == 0]
        prefilling = [(seq_id, n) for seq_id, n in left.items() if n > 0]
        candidates = [(r.id, r.prompt_tokens) for r in waiting]
        prefill, decode = checked_plan(plan, running, prefilling, candidates, max_batch_tokens,
                                       long_prefill_threshold)  # fmt: skip

        # A waiting request that gets a chunk is admitted; only the head of the line can be.
        chunked = {seq_id for seq_id, _ in prefill}
        while waiting and waiting[0].id in chunked:
            engine.admit(waiting.popleft())
        await engine.step(prefill=prefill, decode=decode)


def horizon(requests):
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    return requests[-1].arrival_s + 2 * serial + 1000


def simulate(ctx, plan, requests, *, budget, threshold):
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_batch_tokens=budget)
    queue = asyncio.Queue()
    workloads.drive(sim, requests, queue.put_nowait)
    sim.spawn(serve(engine, queue, plan, budget, threshold))
    run = Run(sim, engine, requests, budget, threshold)
    try:
        sim.run(until=horizon(requests))
    except ProtocolViolation:
        pass  # reported below, with the timeline
    if engine.violations:
        first = engine.violations[0]
        check.that(
            False,
            f"the plan broke the engine's protocol ({first.code})",
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


def expect_steps(run, expected):
    got = run.steps
    detail = "expected steps (start ms, prefill, decode):\n"
    detail += "".join(f"  {step!r}\n" for step in expected[:40])
    detail += "got:\n" + "".join(f"  {step!r}\n" for step in got[:40]) + run.timeline()
    for k, (have, want) in enumerate(zip(got, expected, strict=False)):
        check.that(have[2] == want[2], f"step {k} decodes the wrong sequences", detail)
        check.that(have[1] == want[1], f"step {k} has the wrong prefill chunks", detail)
        check.that(have[0] == want[0], f"step {k} starts at the wrong time", detail)
    check.that(len(got) == len(expected), "the number of steps is wrong", detail)


def solve(ctx, requests, *, budget, threshold):
    return simulate(ctx, ctx.solution.plan, requests, budget=budget, threshold=threshold)


def matches_oracle(ctx, requests, *, budget, threshold):
    got = solve(ctx, requests, budget=budget, threshold=threshold)
    want = simulate(ctx, ctx.oracle.plan, requests, budget=budget, threshold=threshold)
    expect_steps(got, want.steps)
    return got


def max_gap(times):
    """The longest time between two consecutive tokens of one sequence."""
    return max((b - a for a, b in pairwise(times)), default=0.0)


def long_prompt_scenario():
    """Six chats are mid-generation when a 4,000-token prompt arrives at t=100, with
    three short prompts right behind it."""
    chats = [SimRequest(f"chat-{i}", i, 32, 120) for i in range(6)]
    long = SimRequest("long", 100, 4000, 8)
    shorts = [SimRequest(f"short-{i}", 101 + i, 24, 8) for i in range(3)]
    return [*chats, long, *shorts]


@case("the worked example: decodes first, then chunks in arrival order", public=True)
def worked_example(ctx):
    rows = [(0, 16, 4), (0, 40, 2), (3, 8, 2)]
    run = solve(ctx, workloads.replay(rows), budget=32, threshold=24)
    expect_steps(run, [
        (0, ((0, 16), (1, 16)), ()),
        (16, ((1, 24), (2, 7)), (0,)),
        (32.75, ((2, 1),), (0, 1)),
        (43, (), (0, 2)),
    ])  # fmt: skip
    check.equal(run.sim.trace.latencies("arrive", "first_token"), [16.0, 32.75, 40.0],
                label="time to first token (ms)")  # fmt: skip


@case("plan, called directly: the budget is spent on decodes, then on prompts in order",
      public=True)
def plan_directly(ctx):
    # The second step of the worked example.
    expect_plan(ctx, [0], [(1, 24)], [(2, 8)], 32, 24, ([(1, 24), (2, 7)], [0]))
    # Nothing to prefill; nothing to decode.
    expect_plan(ctx, ["a", "b"], [], [], 32, 24, ([], ["a", "b"]))
    expect_plan(ctx, [], [], [("w", 10)], 32, 24, ([("w", 10)], []))
    # Sequences that are already being prefilled come before waiting requests.
    expect_plan(ctx, [], [("p", 5)], [("w", 6), ("x", 7)], 100, 50,
                ([("p", 5), ("w", 6), ("x", 7)], []))  # fmt: skip
    # The budget runs out in the middle of the line: nobody behind gets a chunk.
    expect_plan(ctx, ["r"], [("p", 5)], [("w", 6), ("x", 7)], 10, 50,
                ([("p", 5), ("w", 4)], ["r"]))  # fmt: skip
    expect_plan(ctx, ["r"], [("p", 5)], [("w", 6), ("x", 7)], 6, 50, ([("p", 5)], ["r"]))


@case("a prompt longer than the budget is split across steps", public=True)
def long_prompt_is_split(ctx):
    # Rejects whole_prompt_in_one_step and waits_until_prompt_fits.
    run = solve(ctx, workloads.replay([(0, 100, 2)]), budget=32, threshold=64)
    expect_steps(run, [
        (0, ((0, 32),), ()),
        (16, ((0, 32),), ()),
        (32, ((0, 32),), ()),
        (48, ((0, 4),), ()),
        (57, (), (0,)),
    ])  # fmt: skip


@case("no sequence gets more than long_prefill_threshold prompt tokens in one step",
      public=True)
def threshold_caps_a_chunk(ctx):
    # Rejects ignores_long_threshold: the long prompt leaves room for the short ones.
    expect_plan(ctx, [], [], [("long", 500), ("s1", 20), ("s2", 30)], 128, 64,
                ([("long", 64), ("s1", 20), ("s2", 30)], []))  # fmt: skip
    expect_plan(ctx, ["r"], [("long", 436)], [("s3", 90), ("s4", 9)], 128, 64,
                ([("long", 64), ("s3", 63)], ["r"]))  # fmt: skip
    # A threshold at or above the budget changes nothing.
    expect_plan(ctx, [], [], [("long", 500), ("s1", 20)], 128, 128, ([("long", 128)], []))
    run = solve(ctx, workloads.replay([(0, 200, 1), (0, 16, 1), (1, 16, 2)]), budget=96,
                threshold=64)  # fmt: skip
    expect_steps(run, [
        (0, ((0, 64), (1, 16)), ()),
        (28, ((0, 64), (2, 16)), ()),
        (56, ((0, 64),), (2,)),
        (81, ((0, 8),), ()),
    ])  # fmt: skip


@case("a sequence is decoded only from the step after its last prompt chunk")
def no_decode_before_prefill_is_complete(ctx):
    # Rejects decodes_final_chunk_early: "p" finishes its prompt here and is not decoded.
    expect_plan(ctx, ["r"], [("p", 3)], [], 16, 8, ([("p", 3)], ["r"]))
    expect_plan(ctx, [], [("p", 3)], [("w", 2)], 16, 8, ([("p", 3), ("w", 2)], []))
    run = solve(ctx, workloads.replay([(0, 12, 3), (0, 4, 2)]), budget=8, threshold=8)
    expect_steps(run, [
        (0, ((0, 8),), ()),
        (10, ((0, 4), (1, 4)), ()),
        (20, (), (0, 1)),
        (30, (), (0,)),
    ])  # fmt: skip


@case("decodes count against the budget, and a tiny budget still makes progress")
def decodes_are_charged(ctx):
    # Rejects decodes_not_charged.
    expect_plan(ctx, [1, 2, 3], [(4, 50)], [], 8, 8, ([(4, 5)], [1, 2, 3]))
    expect_plan(ctx, [1, 2, 3], [], [(4, 50)], 3, 8, ([], [1, 2, 3]))
    # More running sequences than budget: the oldest max_batch_tokens of them decode.
    expect_plan(ctx, [1, 2, 3, 4, 5], [(6, 9)], [(7, 9)], 3, 8, ([], [1, 2, 3]))
    # A budget of one token: one decode, or one prompt token, per step.
    expect_plan(ctx, [], [], [("w", 3)], 1, 1, ([("w", 1)], []))
    run = solve(ctx, workloads.replay([(0, 2, 2)]), budget=1, threshold=4)
    expect_steps(run, [(0, ((0, 1),), ()), (8.25, ((0, 1),), ()), (16.5, (), (0,))])


@case("first come, first served: a long prompt is not overtaken while it still gets a chunk")
def first_come_first_served(ctx):
    # Rejects shortest_prompt_first.
    expect_plan(ctx, [], [("old", 300)], [("new", 5)], 64, 64, ([("old", 64)], []))
    expect_plan(ctx, [], [], [("a", 40), ("b", 10), ("c", 10)], 48, 48,
                ([("a", 40), ("b", 8)], []))  # fmt: skip


@case("random states: plan agrees with the reference")
def random_plans(ctx):
    rng = ctx.rng
    for trial in range(300):
        budget = int(rng.integers(1, 80))
        threshold = int(rng.integers(1, 100))
        ids = [f"s{j}" for j in rng.permutation(24)]
        n_run, n_pre, n_wait = (int(n) for n in rng.integers(0, 7, size=3))
        running = ids[:n_run]
        prefilling = [(i, int(rng.integers(1, 120))) for i in ids[n_run:n_run + n_pre]]
        waiting = [(i, int(rng.integers(1, 120))) for i in ids[n_run + n_pre:n_run + n_pre + n_wait]]
        got = checked_plan(ctx.solution.plan, running, prefilling, waiting, budget, threshold)
        want = ctx.oracle.plan(list(running), list(prefilling), list(waiting), budget, threshold)
        detail = (
            f"trial {trial}: plan(running={running!r}, prefilling={prefilling!r}, "
            f"waiting={waiting!r}, max_batch_tokens={budget}, "
            f"long_prefill_threshold={threshold})\nexpected {want!r}\ngot {got!r}\n"
        )
        check.that(got[1] == want[1], "plan returned the wrong decode list", detail)
        check.that(got[0] == want[0], "plan returned the wrong prefill chunks", detail)


@case("random traffic with prompts far longer than the budget")
def random_workloads(ctx):
    for budget, threshold, rate in ((64, 32, 0.01), (256, 96, 0.02), (512, 512, 0.03),
                                    (48, 200, 0.008)):  # fmt: skip
        requests = workloads.poisson(ctx.rng, rate=rate, n=60, prompt_tokens=(1, 700),
                                     output_tokens=(1, 24), quantum_s=1)
        run = matches_oracle(ctx, requests, budget=budget, threshold=threshold)
        check.that(
            max(e["tokens"] for e in run.sim.trace.of("step")) <= budget,
            "a step exceeded max_batch_tokens",
            run.timeline(),
        )


@case("a 4,000-token prompt arrives: running requests keep their token cadence")
def long_prompt_does_not_stall_decodes(ctx):
    # Rejects prefill_first. The threshold equals the budget, so the long prompt may
    # take everything the decodes leave: 506 tokens, a step of 8 + 126.5 + 6 = 140.5 ms.
    requests = long_prompt_scenario()
    run = solve(ctx, requests, budget=512, threshold=512)
    times = run.token_times()
    gap = max(max_gap(times[f"chat-{i}"]) for i in range(6))
    check.at_most("max_inter_token_gap", gap, 150.0, unit="ms")


@case("with a threshold, short prompts behind the long one get their first token quickly")
def short_prompts_are_not_stuck(ctx):
    # Rejects ignores_long_threshold on latency, not only on the chunk sizes.
    requests = long_prompt_scenario()
    run = solve(ctx, requests, budget=512, threshold=128)
    arrival = {r.id: r.arrival_s for r in requests}
    first = {e["id"]: e.time for e in run.sim.trace.of("first_token")}
    ttft = max(first[f"short-{i}"] - arrival[f"short-{i}"] for i in range(3))
    check.at_most("short_prompt_ttft", ttft, 120.0, unit="ms")
    gap = max(max_gap(run.token_times()[f"chat-{i}"]) for i in range(6))
    check.at_most("max_inter_token_gap", gap, 150.0, unit="ms")
