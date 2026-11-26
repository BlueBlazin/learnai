"""Tests for preemption-recompute.

`run` drives a `SimEngine` whose KV cache is a pool of blocks, on a clock that counts
milliseconds. The test records every engine step and every preemption and checks
them three ways: against literal schedules on small scenarios, against rules computed
here from the statement (no protocol violation, every token produced exactly once,
the victim is the newest sequence, a step that preempts admits nothing), and against
the oracle's trace on seeded workloads. The cost of the policy, prompt and output
tokens computed a second time, is bounded on a workload that keeps the pool full.
"""

import asyncio

from learnai.judge import case, check
from learnai.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    def __init__(self, sim, engine, requests, config):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.config = config

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [
            (e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")
        ]

    @property
    def preempts(self):
        """[(time, id)]."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("preempt")]

    def timeline(self):
        max_running, kv_blocks, block_tokens, watermark = self.config
        lines = [
            (
                f"max_running={max_running}, kv_blocks={kv_blocks}, block_tokens={block_tokens}, "
                f"watermark_blocks={watermark}; a step takes 8 + 0.25 * P + D ms"
            ),
            "timeline:",
        ]
        kinds = ("arrive", "step", "finish", "preempt", "release", "violation")
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:100]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "violation":
                what = f"VIOLATION {e['code']}: {e['message']}"
            elif e.kind == "preempt":
                what = f"preempt  {e['id']!r} (after {e['generated']} tokens)"
            else:
                what = f"{e.kind:<8} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 100:
            lines.append(f"  ... {len(shown) - 100} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, module, requests, *, max_running, kv_blocks, block_tokens=16, watermark=0):
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, kv_blocks=kv_blocks, block_tokens=block_tokens)
    queue = asyncio.Queue()
    run = Run(sim, engine, requests, (max_running, kv_blocks, block_tokens, watermark))
    workloads.drive(sim, requests, queue.put_nowait)
    sim.spawn(module.run(engine, queue, max_running, watermark))
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    try:
        # Generous: every request alone, twice over, and room for recomputation.
        sim.run(until=requests[-1].arrival_s + 20 * serial + 1000)
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
    max_running = run.config[0]
    trace = run.sim.trace
    by_id = {r.id: r for r in run.requests}

    finished = {e["id"] for e in trace.of("finish")}
    missing = [r.id for r in run.requests if r.id not in finished]
    check.that(
        not missing,
        f"{len(missing)} of {len(run.requests)} requests never finished",
        detail,
    )
    check.that(not trace.of("release"), "a sequence was released; preemption keeps it", detail)

    # Every token index exactly once: a preempted sequence continues, it does not restart.
    produced = {}
    for e in trace.of("token"):
        produced.setdefault(e["id"], []).append(e["index"])
    for request in run.requests:
        got = produced.get(request.id, [])
        check.that(
            got == list(range(request.output_tokens)),
            "a request's tokens were not produced exactly once each, in order",
            detail + f"request {request.id!r}: token indices {got}, expected "
            f"0..{request.output_tokens - 1}\n",
        )  # fmt: skip
    admits = [e["id"] for e in trace.of("admit")]
    check.that(
        len(admits) == len(set(admits)),
        "a request was admitted to the engine more than once",
        detail,
    )

    # Replay the steps: who holds blocks, in what order they were (re)admitted.
    holders = []  # ids, oldest (re)admission first
    events = [e for e in trace.events if e.kind in ("step", "preempt", "finish", "arrive")]
    arrival_order = []
    waiting_since = {}
    preempted_now = []
    for e in events:
        if e.kind == "arrive":
            arrival_order.append(e["id"])
            waiting_since[e["id"]] = e.time
        elif e.kind == "preempt":
            check.that(
                holders and holders[-1] == e["id"],
                "the preempted sequence was not the most recently admitted running one",
                detail + f"at t={e.time:g}: preempted {e['id']!r}; running, oldest first: "
                f"{holders}\n",
            )  # fmt: skip
            holders.pop()
            preempted_now.append(e.time)
        elif e.kind == "finish":
            if e["id"] in holders:
                holders.remove(e["id"])
        else:
            prefill = [i for i, _ in e["prefill"]]
            check.that(
                list(e["decode"]) == holders,
                "a step must decode exactly the sequences that hold memory, oldest first",
                detail + f"step {e['index']} at t={e.time:g}: decode={list(e['decode'])}, "
                f"running: {holders}\n",
            )  # fmt: skip
            check.that(
                not (prefill and e.time in preempted_now),
                "a step that preempted a sequence also admitted one",
                detail + f"step {e['index']} at t={e.time:g}\n",
            )
            holders.extend(prefill)
            check.that(
                len(holders) <= max_running,
                "more than max_running sequences hold memory",
                detail + f"step {e['index']} at t={e.time:g}: {holders}\n",
            )
    return by_id


def check_literal(run, steps, preempts):
    detail = run.timeline() + "expected steps (start, prefill, decode):\n"
    detail += "".join(f"  {s!r}\n" for s in steps) + f"expected preemptions: {preempts!r}\n"
    got = run.steps
    check.that(
        len(got) == len(steps),
        f"the scheduler ran {len(got)} steps, expected {len(steps)}",
        detail,
    )
    for index, (mine, want) in enumerate(zip(got, steps, strict=True)):
        check.that(
            mine == want,
            f"step {index}: got start={mine[0]:g}, prefill={list(mine[1])}, "
            f"decode={list(mine[2])}; the policy prescribes start={want[0]:g}, "
            f"prefill={list(want[1])}, decode={list(want[2])}",
            detail,
        )
    check.that(
        run.preempts == preempts,
        f"the preemptions (time, id) were {run.preempts!r}, expected {preempts!r}",
        detail,
    )


def check_against_oracle(ctx, requests, label, **config):
    want = simulate(ctx, ctx.oracle, requests, **config)
    got = simulate(ctx, ctx.solution, requests, **config)
    check_rules(got)
    mine, theirs = got.steps, want.steps
    first = next((i for i, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b), None)
    if first is None and len(mine) != len(theirs):
        first = min(len(mine), len(theirs))
    if first is not None:
        a = mine[first] if first < len(mine) else None
        b = theirs[first] if first < len(theirs) else None
        check.that(
            False,
            f"{label}: step {first} is {a!r}, the policy prescribes {b!r}",
            got.timeline() + f"step {first}: got {a!r}\n         expected {b!r}\n",
        )
    check.that(
        got.preempts == want.preempts,
        f"{label}: the preemptions differ from the ones the policy prescribes",
        got.timeline() + f"expected (time, id): {want.preempts[:20]!r}\n",
    )
    return got


EXAMPLE = [(0, 8, 6), (0, 8, 4), (4, 6, 5), (30, 4, 2)]


@case("the worked example: 7 blocks of 4 tokens, one preemption, one resumption",
      public=True)  # fmt: skip
def worked_example(ctx):
    run = simulate(ctx, ctx.solution, workloads.replay(EXAMPLE), max_running=3, kv_blocks=7,
                   block_tokens=4, watermark=1)  # fmt: skip
    check_literal(
        run,
        [
            (0, ((0, 8), (1, 8)), ()),
            (12, (), (0, 1)),
            (22, (), (0, 1)),
            (32, (), (0, 1)),
            (42, ((2, 6), (3, 4)), (0,)),
            (53.5, (), (0, 2)),
            (63.5, ((3, 5),), (2,)),
            (73.75, (), (2,)),
            (82.75, (), (2,)),
        ],
        [(53.5, 3)],
    )
    check_rules(run)
    check.cost(run.engine.counters, "recomputed_tokens", exactly=5, unit="tokens")


@case("the newest running sequence is the victim, and it returns ahead of the queue",
      public=True)  # fmt: skip
def victim_and_return(ctx):
    # Rejects preempts_oldest, victim_to_back, restarts_from_scratch.
    # 6 blocks of 4. Requests 0, 1, 2 fill the pool; request 0 needs a block first.
    requests = workloads.replay([(0, 4, 9), (0, 8, 3), (0, 8, 3), (1, 4, 2), (2, 4, 2)])
    run = simulate(ctx, ctx.solution, requests, max_running=3, kv_blocks=6, block_tokens=4)
    check_literal(
        run,
        [
            (0, ((0, 4), (1, 8), (2, 8)), ()),  # 1 + 2 + 2 blocks; 1 free
            (13, (), (0, 1)),  # 0 takes the free block; 1 needs one: 2 is preempted
            (23, (), (0, 1)),  # 1 finishes with 3 tokens; nothing was admitted at 13
            (33, ((2, 9), (3, 4)), (0,)),  # 2 is back first, with 8 + 1 tokens; then 3
            (45.25, (), (0, 2)),  # 3 needs a block, none is free, and 3 is the newest
            (55.25, ((3, 5), (4, 4)), (0,)),  # 2 has finished; 3 returns ahead of 4
            (66.5, (), (0, 4)),
            (76.5, (), (0,)),
            (85.5, (), (0,)),
        ],
        [(13, 2), (45.25, 3)],
    )
    check_rules(run)


@case("enough memory: no preemption, the steps of plain continuous batching", public=True)
def no_pressure(ctx):
    # Rejects eager preemption and watermark_on_decode.
    requests = workloads.replay(
        [(0, 8, 4), (0, 16, 2), (5, 8, 3), (45, 4, 2), (50, 12, 1), (60, 4, 2)]
    )
    run = simulate(ctx, ctx.solution, requests, max_running=2, kv_blocks=64, block_tokens=16,
                   watermark=2)  # fmt: skip
    check_literal(
        run,
        [
            (0, ((0, 8), (1, 16)), ()),
            (14, (), (0, 1)),
            (24, ((2, 8),), (0,)),
            (35, (), (0, 2)),
            (45, ((3, 4),), (2,)),
            (55, ((4, 12),), (3,)),
            (67, ((5, 4),), ()),
            (76, (), (5,)),
        ],
        [],
    )
    check_rules(run)


@case("the watermark holds back admission next to running sequences, not a lone request",
      public=True)  # fmt: skip
def watermark_rules(ctx):
    # Rejects no_watermark, watermark_always, admission_ignores_decode_growth.
    # 4 blocks of 4. Request 0 fills 3 of them; request 1 (1 block) would fit.
    requests = workloads.replay([(0, 12, 3), (0, 4, 2), (0, 16, 1)])
    run = simulate(ctx, ctx.solution, requests, max_running=4, kv_blocks=4, block_tokens=4,
                   watermark=1)  # fmt: skip
    check_literal(
        run,
        [
            (0, ((0, 12),), ()),  # 1: 4 - 3 - 1 = 0 free would be left, below the watermark
            (11, (), (0,)),  # 0 takes the last block; 1 no longer fits at all
            (20, (), (0,)),
            (29, ((1, 4),), ()),  # alone in the step: the watermark does not apply
            (38, (), (1,)),
            (47, ((2, 16),), ()),  # needs the whole pool; alone, so it is admitted
        ],
        [],
    )
    check_rules(run)


def pressure_workload(ctx, n=120):
    return workloads.poisson(
        ctx.rng, rate=0.04, n=n, prompt_tokens=(16, 160), output_tokens=(8, 96), quantum_s=0.25
    )


@case("a pool that is always nearly full: recomputed tokens at most 3000", public=True)
def recompute_bound(ctx):
    # Rejects no_watermark (admits into the last blocks and preempts at once).
    requests = pressure_workload(ctx)
    run = simulate(ctx, ctx.solution, requests, max_running=8, kv_blocks=40, watermark=4)
    check_rules(run)
    ctx.metric("preemptions", run.engine.counters.preemptions)
    check.cost(run.engine.counters, "recomputed_tokens", at_most=3000, unit="tokens")


@case("a sequence that cannot get a block while it is the newest preempts itself")
def preempts_itself(ctx):
    # 3 blocks of 4: request 0 (2 blocks) and request 1 (1 block) fill the pool.
    requests = workloads.replay([(0, 7, 4), (0, 4, 3), (0, 4, 1)])
    run = simulate(ctx, ctx.solution, requests, max_running=3, kv_blocks=3, block_tokens=4)
    want = simulate(ctx, ctx.oracle, requests, max_running=3, kv_blocks=3, block_tokens=4)
    check_literal(run, want.steps, want.preempts)
    check.that(
        (10.75, 1) in run.preempts,
        "request 1 needed a block at t=10.75 with none free and nothing newer to preempt",
        run.timeline(),
    )
    check_rules(run)


@case("several victims in one step come back in their old order; ties at step ends")
def several_victims(ctx):
    # Rejects victim_to_back (the order of two victims).
    requests = workloads.bursty(
        ctx.rng, bursts=6, burst_size=6, period_s=40.0, prompt_tokens=(3, 4),
        output_tokens=(6, 14),
    )  # fmt: skip
    got = check_against_oracle(ctx, requests, "bursts of six", max_running=6, kv_blocks=9,
                               block_tokens=4)  # fmt: skip
    twice = {t for t, _ in got.preempts if sum(1 for u, _ in got.preempts if u == t) > 1}
    check.that(bool(twice), "this workload must preempt two sequences in one step", got.timeline())


@case("bursty and Poisson workloads under memory pressure: the trace equals the policy's")
def exact_traces(ctx):
    bursts = workloads.bursty(
        ctx.rng, bursts=8, burst_size=7, period_s=250.0, prompt_tokens=(8, 100),
        output_tokens=(2, 60),
    )  # fmt: skip
    for watermark in (0, 2, 6):
        check_against_oracle(ctx, bursts, f"bursts, watermark {watermark}", max_running=6,
                             kv_blocks=24, watermark=watermark)  # fmt: skip
    steady = pressure_workload(ctx, n=100)
    check_against_oracle(ctx, steady, "Poisson, 8 slots", max_running=8, kv_blocks=40,
                         watermark=4)  # fmt: skip
    check_against_oracle(ctx, steady, "Poisson, 3 slots, small blocks", max_running=3,
                         kv_blocks=70, block_tokens=5, watermark=1)  # fmt: skip
    check_against_oracle(ctx, steady, "Poisson, memory to spare", max_running=4,
                         kv_blocks=400, watermark=8)  # fmt: skip


@case("requests as large as the whole pool, and a pool of one-token blocks")
def extremes(ctx):
    rows = [(0.0, 30, 3), (0.0, 2, 30), (1.0, 20, 13), (2.0, 1, 1), (2.0, 32, 1), (3.0, 16, 17)]
    requests = [SimRequest(i, *row) for i, row in enumerate(rows)]
    check_against_oracle(ctx, requests, "contexts of exactly 32 tokens in 8 blocks of 4",
                         max_running=4, kv_blocks=8, block_tokens=4, watermark=3)  # fmt: skip
    check_against_oracle(ctx, requests, "32 blocks of one token", max_running=2, kv_blocks=32,
                         block_tokens=1, watermark=0)  # fmt: skip
    check_against_oracle(ctx, requests, "one slot", max_running=1, kv_blocks=8, block_tokens=4,
                         watermark=7)  # fmt: skip
