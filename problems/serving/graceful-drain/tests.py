"""Tests for graceful-drain.

`simulate` plays the clients and the operator: at each arrival it calls
`server.submit(request)` and, if the request is accepted, starts a task that awaits
`handle.result()` and records how the request ended; at `drain_at` it calls
`server.drain(grace_ms)` from a plain timer and records when the returned future
resolves. The engine is a `SimEngine` on a simulated clock whose unit is the
millisecond.

The statement determines everything, so each case checks the outcomes, the engine
steps, the releases, and the drain: against literal values on the hand-checked
scenarios, and on seeded workloads against rules computed here from the statement and
against the oracle.
"""

import asyncio

from learnai.judge import case, check
from learnai.kit import CostModel, ProtocolViolation, SimEngine, SimRequest, workloads

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, late, drain_at, grace_ms, max_running):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.late = late  # ids delivered after the drain call although they share its instant
        self.drain_at = drain_at
        self.grace_ms = grace_ms
        self.max_running = max_running
        self.ready_after_call = None
        self.returned = None  # what kind of object drain() returned
        self.submit_after_close = None

    @property
    def deadline(self):
        return self.drain_at + max(self.grace_ms, 0)

    @property
    def steps(self):
        """[(start time, prefill, decode)]."""
        return [
            (e.time, tuple(e["prefill"]), tuple(e["decode"])) for e in self.sim.trace.of("step")
        ]

    @property
    def releases(self):
        """[(time, id)] of `engine.release` calls."""
        return [(e.time, e["id"]) for e in self.sim.trace.of("release")]

    @property
    def outcomes(self):
        """{id: (time, how, tokens)}; how is "finished", "terminated", or "refused"."""
        return {e["id"]: (e.time, e["how"], e["tokens"]) for e in self.sim.trace.of("outcome")}

    @property
    def drained(self):
        """(time, value, server.closed at that moment) of the drain future, or None."""
        events = self.sim.trace.of("drained")
        return (events[0].time, events[0]["value"], events[0]["closed"]) if events else None

    def timeline(self):
        """A readable log of the simulation, for failure details."""
        header = f"max_running={self.max_running}; a step takes 8 + 0.25 * P + D ms; "
        header += f"drain({self.grace_ms:g}) is called at t={self.drain_at:g}"
        lines = [header, "timeline:"]
        kinds = ("arrive", "step", "token", "finish", "release", "outcome", "drain", "drained",
                 "run_returned")  # fmt: skip
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:90]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "token":
                what = f"  token  {e['id']!r} #{e['index']}"
            elif e.kind == "outcome":
                what = f"client   {e['id']!r}: {e['how']}"
                what += f" with {e['tokens']} tokens" if e["how"] != "refused" else ""
            elif e.kind == "drain":
                what = f"drain({self.grace_ms:g}) called; server.ready is then {e['ready']!r}"
            elif e.kind == "drained":
                what = f"the drain future resolves to {e['value']!r}; server.closed={e['closed']!r}"
            elif e.kind == "run_returned":
                what = "run() returns"
            else:
                what = f"{e.kind:<8} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 90:
            lines.append(f"  ... {len(shown) - 90} more events")
        return "\n".join(lines) + "\n"


def horizon(requests, drain_at, grace_ms):
    """A time by which any server that keeps working has ended every request."""
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    last = max([drain_at + grace_ms, *(r.arrival_s for r in requests)])
    return last + 2 * serial + 1000


def simulate(ctx, module, requests, *, drain_at, grace_ms, max_running, late=()):
    """Run `module.Server` over `requests` and drain it at `drain_at`.

    Arrivals at the drain instant are delivered before the drain call, except those
    whose id is in `late`, which are delivered right after it.
    """
    sim = ctx.simclock()
    engine = SimEngine(sim, COST, max_seqs=max_running)
    server = module.Server(engine, max_running)
    run = Run(sim, engine, requests, set(late), drain_at, grace_ms, max_running)

    async def client(request, handle):
        try:
            tokens = await handle.result()
            how = "finished"
        except module.Terminated as error:
            tokens, how = getattr(error, "tokens", None), "terminated"
        sim.trace.record("outcome", id=request.id, how=how, tokens=tokens, closed=server.closed)

    def arrive(request):
        try:
            handle = server.submit(request)
        except module.Unavailable:
            sim.trace.record("outcome", id=request.id, how="refused", tokens=None,
                             closed=server.closed)  # fmt: skip
            return
        sim.spawn(client(request, handle))

    def drain():
        returned = server.drain(grace_ms)
        run.ready_after_call = server.ready
        run.returned = type(returned).__name__
        sim.trace.record("drain", ready=server.ready)
        if asyncio.isfuture(returned):
            returned.add_done_callback(
                lambda f: sim.trace.record("drained", value=f.result(), closed=server.closed)
            )
        elif asyncio.iscoroutine(returned):
            returned.close()

    async def serve():
        await server.run()
        sim.trace.record("run_returned")

    def knock():
        try:
            server.submit(SimRequest("after-close", sim.now(), 8, 1))
        except module.Unavailable:
            run.submit_after_close = "Unavailable"
        else:
            run.submit_after_close = "accepted"

    workloads.drive(sim, [r for r in requests if r.id not in run.late], arrive)
    sim.call_at(drain_at, drain)
    workloads.drive(sim, [r for r in requests if r.id in run.late], arrive)
    sim.spawn(serve())
    end = horizon(requests, drain_at, grace_ms)
    try:
        sim.run(until=end)
        if not engine.violations:
            sim.call_at(end + 1, knock)
            sim.run(until=end + 2)
    except ProtocolViolation:
        pass  # reported below, with the timeline
    if engine.violations:
        first = engine.violations[0]
        check.that(
            False,
            f"the server broke the engine's protocol ({first.code})",
            f"{first.code}: {first}\n" + run.timeline(),
        )
    return run


def check_rules(run):
    """The statement, restated as properties of the trace (independent of the oracle)."""
    detail = run.timeline()
    trace = run.sim.trace
    check.that(
        run.returned in ("Future", "Task"),
        "drain() must be a plain method that returns an asyncio future",
        f"it returned a {run.returned}\n" + detail,
    )
    check.that(
        run.ready_after_call is False,
        "server.ready is not False when drain() returns",
        f"server.ready was {run.ready_after_call!r} right after the call\n" + detail,
    )
    outcomes = run.outcomes
    missing = [r.id for r in run.requests if r.id not in outcomes]
    check.that(
        not missing,
        f"{len(missing)} of {len(run.requests)} clients are still waiting for `result()` "
        "at the end of the simulation",
        f"no outcome for: {missing[:10]!r}\n" + detail,
    )

    finish_time = {e["id"]: e.time for e in trace.of("finish")}
    produced, last_token = {}, {}
    for e in trace.of("token"):
        produced[e["id"]] = produced.get(e["id"], 0) + 1
        last_token[e["id"]] = e.time
    terminated = 0
    for r in run.requests:
        when, how, tokens = outcomes[r.id]
        who = f"request {r.id!r}: "
        early = r.arrival_s < run.drain_at or (r.arrival_s == run.drain_at and r.id not in run.late)
        if not early:
            check.that(how == "refused", "a request was accepted after drain() had been called",
                       f"{who}{how}\n" + detail)  # fmt: skip
            continue
        check.that(how != "refused", "a request that arrived before the drain was refused",
                   who + "\n" + detail)  # fmt: skip
        if how == "finished":
            check.that(r.id in finish_time, "a client was told its request finished, but the "
                       "engine never finished it", who + "\n" + detail)  # fmt: skip
            check.that(when == finish_time[r.id], "result() did not return at the instant of "
                       "the last token", who + "\n" + detail)  # fmt: skip
            check.that(tokens == r.output_tokens, "result() returned the wrong token count",
                       f"{who}returned {tokens!r}\n" + detail)  # fmt: skip
            check.that(when < run.deadline or when < run.drain_at, "a request finished at or "
                       "after the grace deadline", who + "\n" + detail)  # fmt: skip
        else:
            terminated += 1
            check.that(
                when == run.deadline,
                "a pending request was not terminated at the grace deadline",
                f"{who}terminated at t={when:g}; the deadline is t={run.deadline:g}\n" + detail,
            )
            check.that(
                r.id not in last_token or last_token[r.id] < when,
                "the engine produced a token for a request at or after the instant it was "
                "terminated",
                f"{who}terminated at t={when:g}, last token at t={last_token.get(r.id, 0):g}\n"
                + detail,
            )
            check.that(
                tokens == produced.get(r.id, 0),
                "Terminated.tokens is not the number of tokens the request had produced",
                f"{who}Terminated.tokens={tokens!r}, produced {produced.get(r.id, 0)}\n" + detail,
            )

    check.that(run.drained is not None, "the future returned by drain() never resolved", detail)
    resolved_at, value, closed = run.drained
    accepted_ends = [when for when, how, _ in outcomes.values() if how != "refused"]
    idle_from = max([run.drain_at, *accepted_ends, *trace.times("step_done")])
    check.that(
        resolved_at >= max([run.drain_at, *accepted_ends]),
        "the drain resolved while requests were still pending",
        f"resolved at t={resolved_at:g}\n" + detail,
    )
    check.that(
        resolved_at >= idle_from,
        "the drain resolved while the engine was still running a step",
        f"resolved at t={resolved_at:g}; the engine was busy until t={idle_from:g}\n" + detail,
    )
    check.that(
        resolved_at == idle_from,
        "the drain did not resolve at the instant the server became idle",
        f"resolved at t={resolved_at:g}; nothing was pending and the engine was idle from "
        f"t={idle_from:g}\n" + detail,
    )
    check.that(
        type(value) is int and value == terminated,
        "the drain future's value is not the number of terminated requests",
        f"value {value!r}; {terminated} requests were terminated\n" + detail,
    )
    check.that(closed is True, "server.closed is not True when the drain resolves", detail)
    seen_closed = [e["id"] for e in trace.of("outcome") if e["closed"] and e.time < resolved_at]
    check.that(
        not seen_closed,
        "server.closed was True before the drain had resolved",
        f"seen at the outcome of {seen_closed[:5]!r}\n" + detail,
    )
    returned = trace.times("run_returned")
    check.that(
        returned == [resolved_at],
        "run() did not return at the instant the drain resolved",
        f"run() returned at {returned!r}; the drain resolved at t={resolved_at:g}\n" + detail,
    )
    check.that(
        run.submit_after_close == "Unavailable",
        "submit() after the drain has completed must raise Unavailable",
        detail,
    )
    check.that(
        not run.engine.live and not run.engine.busy,
        "the engine still holds sequences at the end of the simulation",
        f"live: {run.engine.live!r}\n" + detail,
    )


def expect(run, *, outcomes=None, steps=None, releases=None, drained=None):
    """Compare with literal values, without leaking them in the message."""
    detail = run.timeline()
    if outcomes is not None:
        got = run.outcomes
        for request_id, want in outcomes.items():
            have = got[request_id]
            line = f"request {request_id!r}: expected (time, how, tokens) {want!r}, got {have!r}\n"
            check.that(have[1] == want[1], "a request ended the wrong way", line + detail)
            check.that(have[0] == want[0], "a client got its result at the wrong time",
                       line + detail)  # fmt: skip
            check.that(have[2] == want[2], "a client was given the wrong token count",
                       line + detail)  # fmt: skip
    if releases is not None:
        # Releases at one instant may come in any order.
        def canonical(pairs):
            return sorted((when, repr(seq_id)) for when, seq_id in pairs)

        check.that(
            canonical(run.releases) == canonical(releases),
            "engine.release was not called for the right sequences at the right times",
            f"expected (time, id): {releases!r}\ngot: {run.releases!r}\n" + detail,
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
    if drained is not None:
        have = run.drained[:2]
        line = f"expected the drain to resolve at (time, value) {drained!r}, got {have!r}\n"
        check.that(have[0] == drained[0], "the drain resolved at the wrong time", line + detail)
        check.that(have[1] == drained[1], "the drain resolved to the wrong value", line + detail)


def solve(ctx, requests, **options):
    run = simulate(ctx, ctx.solution, requests, **options)
    check_rules(run)
    return run


def matches_oracle(ctx, requests, **options):
    got = solve(ctx, requests, **options)
    want = simulate(ctx, ctx.oracle, requests, **options)
    expect(got, outcomes=want.outcomes, releases=want.releases, steps=want.steps,
           drained=want.drained[:2])  # fmt: skip
    return got


def req(id, arrival, prompt, output):
    return SimRequest(id, arrival, prompt, output)


@case("the worked example: a drain with a waiting request, a refusal, and a termination",
      public=True)  # fmt: skip
def worked_example(ctx):
    requests = [req(0, 0, 8, 4), req(1, 0, 8, 2), req(2, 4, 8, 4), req(3, 30, 8, 2)]
    run = solve(ctx, requests, drain_at=20, grace_ms=30, max_running=2)
    expect(
        run,
        outcomes={0: (43, "finished", 4), 1: (22, "finished", 2), 2: (50, "terminated", 2),
                  3: (30, "refused", None)},
        releases=[(50, 2)],
        steps=[(0, ((0, 8), (1, 8)), ()), (12, (), (0, 1)), (22, ((2, 8),), (0,)),
               (33, (), (0, 2)), (43, (), (2,))],
        drained=(52, 1),
    )  # fmt: skip


@case("draining an idle server resolves at once, and later requests are refused", public=True)
def drain_while_idle(ctx):
    # Rejects always_waits_full_grace and run_never_returns.
    requests = [req("a", 0, 8, 2), req("b", 120, 8, 1)]
    run = solve(ctx, requests, drain_at=100, grace_ms=50, max_running=2)
    expect(
        run,
        outcomes={"a": (19, "finished", 2), "b": (120, "refused", None)},
        releases=[],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",))],
        drained=(100, 0),
    )


@case("waiting requests are still served, and the drain resolves when the last one finishes",
      public=True)  # fmt: skip
def everyone_finishes_within_grace(ctx):
    # Rejects waiting_dropped_at_once, resolves_immediately, and always_waits_full_grace:
    # "b" and "c" are waiting when drain(200) is called at t=5 and finish long before 205.
    requests = [req("a", 0, 8, 3), req("b", 2, 16, 2), req("c", 3, 8, 2)]
    run = solve(ctx, requests, drain_at=5, grace_ms=200, max_running=1)
    expect(
        run,
        outcomes={"a": (28, "finished", 3), "b": (49, "finished", 2), "c": (68, "finished", 2)},
        releases=[],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",)), (28, (("b", 16),), ()),
               (40, (), ("b",)), (49, (("c", 8),), ()), (59, (), ("c",))],
        drained=(68, 0),
    )  # fmt: skip


@case("at the grace deadline running and waiting requests are terminated, mid-step",
      public=True)  # fmt: skip
def deadline_cuts_running_and_waiting(ctx):
    # Rejects not_released_at_deadline and resolves_while_engine_busy: the deadline is
    # t=25, in the middle of the step that runs from 19 to 28.
    requests = [req("a", 0, 8, 6), req("b", 1, 8, 2), req("c", 2, 8, 2)]
    run = solve(ctx, requests, drain_at=15, grace_ms=10, max_running=1)
    expect(
        run,
        outcomes={"a": (25, "terminated", 2), "b": (25, "terminated", 0),
                  "c": (25, "terminated", 0)},
        releases=[(25, "a")],
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",))],
        drained=(28, 3),
    )  # fmt: skip


@case("arrivals at the drain instant: accepted before the call, refused after it", public=True)
def arrivals_at_the_drain_instant(ctx):
    # Rejects still_admits and takes_effect_late: "b" is submitted just before
    # drain() at t=10, "c" just after it in the same instant, "d" one millisecond later.
    requests = [req("a", 0, 8, 3), req("b", 10, 8, 1), req("c", 10, 8, 1), req("d", 11, 8, 1)]
    run = solve(ctx, requests, drain_at=10, grace_ms=100, max_running=2, late=["c"])
    expect(
        run,
        outcomes={"a": (30, "finished", 3), "b": (21, "finished", 1), "c": (10, "refused", None),
                  "d": (11, "refused", None)},
        releases=[],
        steps=[(0, (("a", 8),), ()), (10, (("b", 8),), ("a",)), (21, (), ("a",))],
        drained=(30, 0),
    )  # fmt: skip


@case("a token due at exactly the grace deadline is not produced; a moment earlier it is")
def token_due_at_the_deadline(ctx):
    # Rejects deadline_by_sleeping_task. The last token of "a" is due at t=28.
    run = solve(ctx, [req("a", 0, 8, 3)], drain_at=5, grace_ms=23, max_running=2)
    expect(run, outcomes={"a": (28, "terminated", 2)}, releases=[(28, "a")], drained=(28, 1))
    run = solve(ctx, [req("a", 0, 8, 3)], drain_at=5, grace_ms=23.25, max_running=2)
    expect(run, outcomes={"a": (28, "finished", 3)}, releases=[], drained=(28, 0))
    # The deadline falls on the end of an earlier step: that token is lost too.
    run = solve(ctx, [req("a", 0, 8, 3)], drain_at=5, grace_ms=14, max_running=2)
    expect(run, outcomes={"a": (19, "terminated", 1)}, releases=[(19, "a")],
           steps=[(0, (("a", 8),), ()), (10, (), ("a",))], drained=(19, 1))  # fmt: skip


@case("drain(0): everything pending is terminated inside the call")
def grace_zero(ctx):
    # In the middle of a step: the step still runs to its end, and the drain resolves then.
    requests = [req("a", 0, 8, 3), req("b", 1, 8, 3), req("c", 2, 8, 3)]
    run = solve(ctx, requests, drain_at=14, grace_ms=0, max_running=2)
    expect(
        run,
        outcomes={"a": (14, "terminated", 1), "b": (14, "terminated", 0),
                  "c": (14, "terminated", 0)},
        releases=[(14, "a"), (14, "b")],
        steps=[(0, (("a", 8),), ()), (10, (("b", 8),), ("a",))],
        drained=(21, 3),
    )  # fmt: skip
    # At the instant a step ends: the token due then is not produced (a timer set for
    # "now" would fire after the step has completed).
    run = solve(ctx, [req("a", 0, 8, 3), req("b", 1, 8, 3)], drain_at=19, grace_ms=0,
                max_running=1)  # fmt: skip
    expect(run, outcomes={"a": (19, "terminated", 1), "b": (19, "terminated", 0)},
           releases=[(19, "a")], drained=(19, 2))  # fmt: skip
    # On an idle server.
    run = solve(ctx, [req("a", 0, 8, 1)], drain_at=40, grace_ms=0, max_running=1)
    expect(run, outcomes={"a": (10, "finished", 1)}, releases=[], drained=(40, 0))


@case("a drain before the first request, and one long after the last")
def drain_at_the_edges(ctx):
    requests = [req(i, 4 * i + 1, 8, 2) for i in range(5)]
    run = matches_oracle(ctx, requests, drain_at=0, grace_ms=30, max_running=2)
    check.that(not run.steps, "a server drained before any request arrived ran engine steps",
               run.timeline())  # fmt: skip
    matches_oracle(ctx, requests, drain_at=500, grace_ms=30, max_running=2)


@case("more waiting requests than the grace period can serve")
def long_queue(ctx):
    # Rejects waiting_dropped_at_once (they must be served while the grace period
    # lasts) and not_released_at_deadline (two sequences are running at the deadline).
    requests = [req(i, i, 8, 4) for i in range(12)]
    run = matches_oracle(ctx, requests, drain_at=13, grace_ms=90, max_running=2)
    how = [run.outcomes[i][1] for i in range(12)]
    check.that("finished" in how[2:] and "terminated" in how, "the scenario lost its point")


@case("random traffic, drained at a random moment: outcomes, steps, and the drain")
def random_workloads(ctx):
    for max_running, rate in ((1, 0.01), (3, 0.03), (6, 0.05), (4, 0.05)):
        requests = workloads.poisson(ctx.rng, rate=rate, n=60, prompt_tokens=(4, 96),
                                     output_tokens=(1, 24), quantum_s=1)  # fmt: skip
        for _ in range(3):
            drain_at = float(ctx.rng.integers(0, int(requests[-1].arrival_s) + 50))
            grace_ms = float(ctx.rng.choice([0, 5, 40, 150, 600, 5000]))
            matches_oracle(ctx, requests, drain_at=drain_at, grace_ms=grace_ms,
                           max_running=max_running)  # fmt: skip


@case("bursts: grace deadlines that fall exactly when steps end")
def bursty_workloads(ctx):
    for max_running in (2, 4):
        # With 8-token prompts every step lasts a whole number of milliseconds, so
        # whole-millisecond deadlines often coincide with the end of a step.
        requests = workloads.bursty(ctx.rng, bursts=5, burst_size=6, period_s=96,
                                    prompt_tokens=8, output_tokens=(1, 8), quantum_s=1)  # fmt: skip
        for _ in range(6):
            drain_at = float(ctx.rng.integers(0, 420))
            grace_ms = float(ctx.rng.integers(0, 80))
            matches_oracle(ctx, requests, drain_at=drain_at, grace_ms=grace_ms,
                           max_running=max_running)  # fmt: skip
