"""Tests for streaming-backpressure.

`simulate` plays the transport of every client: at each arrival it calls
`server.submit(request)`, and at the client's scripted instants it calls
`stream.take()` (the client can receive data now) or `stream.close()` (the client
disconnected), from plain timers. The engine is a `SimEngine` on a simulated clock
whose unit is the millisecond.

The statement determines everything, so each case checks the chunks every `take`
returned, the stream states, the engine steps, and the releases: against literal
values on the hand-checked scenarios, and on seeded workloads against rules computed
here from the statement and against the oracle. The number of tokens a stream holds
unread is computed from the trace: tokens the engine produced minus tokens taken.

The engine is handed over as `Engine`, a `SimEngine` behind the kit's `Guarded`: a
server may use its operations and read its state and limits. `engine.sim` (the way to
the trace everything here is judged on), `engine.violations`, and the counters are
refused, so what the engine recorded cannot be rewritten.
"""

import bisect

from learnai.judge import case, check
from learnai.kit import (
    CostModel,
    Guarded,
    ProtocolViolation,
    SimEngine,
    SimRequest,
    check_guard,
    guard,
    workloads,
)

# A step over P prompt tokens and D decoded sequences takes 8 + 0.25 * P + D milliseconds.
COST = CostModel(base_s=8, prefill_s_per_token=0.25, decode_s_per_seq=1)


class Engine(Guarded, SimEngine):
    """The engine a server gets: its operations, state, and limits; not its records."""

    label = "engine"
    refused = Guarded.refused | {"sim", "violations"}


class Client:
    """When one client can receive data (`takes`) and when it disconnects."""

    def __init__(self, takes=(), close_at=None):
        self.takes = sorted(takes)
        self.close_at = close_at


def reader(start, period, until, *, away=()):
    """A client that takes every `period` ms from `start` until `until`, except
    during the half-open intervals `away`."""
    times, t = [], start
    while t < until:
        if not any(a <= t < b for a, b in away):
            times.append(t)
        t += period
    return Client(times)


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, engine, requests, clients, max_running, max_buffered, stall_ms):
        self.sim = sim
        self.engine = engine
        self.requests = requests
        self.clients = clients
        self.max_running = max_running
        self.max_buffered = max_buffered
        self.stall_ms = stall_ms
        self.final = {}  # id -> stream.state at the end

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
    def takes(self):
        """[(time, id, chunk, state after the call)] of `take` and `close` calls; the
        chunk of a `close` is None."""
        return [
            (e.time, e["id"], e["chunk"], e["state"])
            for e in self.sim.trace.events
            if e.kind in ("take", "close")
        ]

    def token_times(self, request_id=None):
        """Token times of one request, or {id: times} for all of them."""
        times = {}
        for e in self.sim.trace.of("token"):
            times.setdefault(e["id"], []).append(e.time)
        return times if request_id is None else times.get(request_id, [])

    def peak_buffered(self):
        """The most tokens any stream held unread at any moment, from the trace."""
        level, peak = {}, 0
        for e in self.sim.trace.events:
            if e.kind == "token":
                level[e["id"]] = level.get(e["id"], 0) + 1
                peak = max(peak, level[e["id"]])
            elif e.kind in ("take", "close", "release"):
                level[e["id"]] = 0
        return peak

    def timeline(self):
        """A readable log of the simulation, for failure details."""
        header = f"max_running={self.max_running}, max_buffered={self.max_buffered}, "
        header += f"stall_ms={self.stall_ms:g}; a step takes 8 + 0.25 * P + D ms"
        lines = [header, "timeline:"]
        kinds = ("arrive", "step", "token", "finish", "release", "take", "close")
        shown = [e for e in self.sim.trace.events if e.kind in kinds]
        for e in shown[:110]:
            if e.kind == "arrive":
                r = e["request"]
                what = f"arrive   {r.id!r} (prompt {r.prompt_tokens}, output {r.output_tokens})"
            elif e.kind == "step":
                what = f"step {e['index']}: prefill={list(e['prefill'])} decode={list(e['decode'])}"
                what += f"  ({e['duration']:g} ms)"
            elif e.kind == "token":
                what = f"  token  {e['id']!r} #{e['index']}"
            elif e.kind == "take":
                what = f"take()   {e['id']!r} -> {list(e['chunk'])!r}; state {e['state']!r}"
            elif e.kind == "close":
                what = f"close()  {e['id']!r}; state {e['state']!r}"
            else:
                what = f"{e.kind:<8} {e['id']!r}"
            lines.append(f"  t={e.time:>8g}  {what}")
        if len(shown) > 110:
            lines.append(f"  ... {len(shown) - 110} more events")
        lines.append(f"final states: {self.final!r}")
        return "\n".join(lines) + "\n"


def horizon(requests, clients, stall_ms):
    """A time by which any server that keeps working has ended every stream."""
    serial = sum(
        COST.step_s(r.prompt_tokens, 0) + r.output_tokens * COST.step_s(0, 1) for r in requests
    )
    last = max(r.arrival_s for r in requests)
    for client in clients.values():
        last = max([last, *client.takes, client.close_at or 0])
    return last + 2 * serial + (len(requests) + 1) * stall_ms + 1000


def simulate(ctx, module, requests, clients, *, max_running, max_buffered, stall_ms):
    """Run `module.Server` over `requests`; `clients` maps a request id to its
    `Client`. A request without one is never read."""
    sim = ctx.simclock()
    engine = Engine(sim, COST, max_seqs=max_running)
    guard(ctx)  # the solution's code runs from here to the end of `sim.run`
    server = module.Server(engine, max_running, max_buffered, stall_ms)
    run = Run(sim, engine, requests, clients, max_running, max_buffered, stall_ms)
    streams = {}

    def take(request_id):
        chunk = streams[request_id].take()
        check.that(
            isinstance(chunk, list) and all(type(i) is int for i in chunk),
            "take() must return a list of ints",
            f"it returned {chunk!r}\n",
        )
        state = streams[request_id].state
        sim.trace.record("take", id=request_id, chunk=tuple(chunk), state=state)

    def close(request_id):
        streams[request_id].close()
        sim.trace.record("close", id=request_id, chunk=None, state=streams[request_id].state)

    def arrive(request):
        streams[request.id] = server.submit(request)
        client = clients.get(request.id, Client())
        # Scheduled at the arrival, so before any stall timer of this stream: at a
        # shared instant the take comes first.
        for when in client.takes:
            if when >= request.arrival_s:
                sim.call_at(when, take, request.id)
        if client.close_at is not None:
            sim.call_at(client.close_at, close, request.id)

    workloads.drive(sim, requests, arrive)
    sim.spawn(server.run())
    try:
        sim.run(until=horizon(requests, clients, stall_ms))
    except ProtocolViolation:
        pass  # reported below, with the timeline
    check_guard()
    run.final = {request_id: stream.state for request_id, stream in streams.items()}
    if engine.violations:
        first = engine.violations[0]
        check.that(
            False,
            f"the server broke the engine's protocol ({first.code})",
            f"{first.code}: {first}\n" + run.timeline(),
        )
    return run


def expected_ends(run):
    """{id: (time, "stalled" | "closed")} as the statement defines them, computed
    from the token times the solution produced and the clients' scripts."""
    by_id = {r.id: r for r in run.requests}
    produced, level, full_since, ended = {}, {}, {}, {}
    events = [e for e in run.sim.trace.events if e.kind in ("token", "take", "close", "finish")]
    finished = set()

    def expire(now, inclusive):
        for request_id, since in list(full_since.items()):
            due = since + run.stall_ms
            if due < now or (inclusive and due == now):
                ended.setdefault(request_id, (due, "stalled"))
                del full_since[request_id]

    for e in events:
        request_id = e["id"]
        # A take or close at the very instant a stall is due comes first and wins;
        # anything the engine does at that instant comes after the stall.
        expire(e.time, inclusive=e.kind in ("token", "finish"))
        if request_id in ended:
            continue
        if e.kind == "token":
            produced[request_id] = produced.get(request_id, 0) + 1
            level[request_id] = level.get(request_id, 0) + 1
            done = produced[request_id] == by_id[request_id].output_tokens
            if level[request_id] == run.max_buffered and not done:
                full_since.setdefault(request_id, e.time)
        elif e.kind == "finish":
            finished.add(request_id)
        elif e.kind == "take":
            level[request_id] = 0
            full_since.pop(request_id, None)
        elif request_id not in finished:  # close on an open stream
            ended[request_id] = (e.time, "closed")
            full_since.pop(request_id, None)
    expire(float("inf"), inclusive=False)
    return ended


def check_rules(run):
    """The statement, restated as properties of the trace (independent of the oracle)."""
    detail = run.timeline()
    trace = run.sim.trace
    finish_time = {e["id"]: e.time for e in trace.of("finish")}
    ends = expected_ends(run)
    released = {}
    for when, request_id in run.releases:
        released.setdefault(request_id, when)

    check.that(
        run.peak_buffered() <= run.max_buffered,
        "a stream held more unread tokens than max_buffered",
        f"peak: {run.peak_buffered()} tokens\n" + detail,
    )
    token_times = run.token_times()
    calls = {}
    for call in run.takes:
        calls.setdefault(call[1], []).append(call)
    for r in run.requests:
        who = f"request {r.id!r}: "
        tokens = token_times.get(r.id, [])
        taken = []
        for when, _, chunk, state in calls.get(r.id, []):
            end = ends.get(r.id)
            over = end is not None and when >= end[0]
            if chunk is None:
                want_state = "finished" if r.id in finish_time and not over else "closed"
                if over and end[1] == "stalled":
                    want_state = "stalled"
            elif over:
                want_state = end[1]
                check.that(not chunk, "take() returned tokens of a stream that had been closed "
                           "or cut off", f"{who}at t={when:g}: {list(chunk)!r}\n" + detail)  # fmt: skip
            else:
                want_state = (
                    "finished" if r.id in finish_time and finish_time[r.id] < when else "open"
                )
                produced = bisect.bisect_left(tokens, when)
                want = list(range(len(taken), produced))
                taken.extend(chunk)
                check.that(
                    list(chunk) == want,
                    "take() did not return exactly the tokens produced since the last take",
                    f"{who}at t={when:g}: got {list(chunk)!r}, expected {want!r}\n" + detail,
                )
            check.that(
                state == want_state,
                "a stream was in the wrong state after a call",
                f"{who}at t={when:g}: state {state!r}, expected {want_state!r}\n" + detail,
            )
        if r.id in ends:
            when, how = ends[r.id]
            check.that(
                run.final.get(r.id) == how,
                "a stream whose buffer stayed full for stall_ms was not cut off"
                if how == "stalled"
                else "a closed stream is not in the state 'closed'",
                f"{who}expected {how!r} at t={when:g}, final state {run.final.get(r.id)!r}\n"
                + detail,
            )
            check.that(
                not tokens or tokens[-1] < when,
                "the engine produced a token for a stream at or after the instant it was "
                "closed or cut off",
                f"{who}{how} at t={when:g}, last token at t={tokens[-1] if tokens else 0:g}\n"
                + detail,
            )
            if tokens or r.id in released:
                check.that(
                    released.get(r.id) == when,
                    "a running sequence was not released at the instant its stream was "
                    "closed or cut off",
                    f"{who}{how} at t={when:g}, released at {released.get(r.id)!r}\n" + detail,
                )
        else:
            check.that(
                run.final.get(r.id) == "finished",
                "a stream that was neither closed nor stalled did not finish",
                f"{who}final state {run.final.get(r.id)!r}\n" + detail,
            )
            check.that(r.id not in released, "a healthy stream was released from the engine",
                       who + "\n" + detail)  # fmt: skip
    check.that(
        not run.engine.live and not run.engine.busy,
        "the engine still holds sequences at the end of the simulation",
        f"live: {run.engine.live!r}\n" + detail,
    )


def expect(run, *, steps=None, releases=None, takes=None, final=None):
    """Compare with literal values, without leaking them in the message."""
    detail = run.timeline()
    if takes is not None:
        got = run.takes
        lines = "expected calls (time, id, chunk, state after):\n"
        lines += "".join(f"  {call!r}\n" for call in takes[:40])
        lines += "got:\n" + "".join(f"  {call!r}\n" for call in got[:40]) + detail
        for have, want in zip(got, takes, strict=True):
            have = (have[0], have[1], None if have[2] is None else list(have[2]), have[3])
            want = (want[0], want[1], None if want[2] is None else list(want[2]), want[3])
            check.that(have[2] == want[2], "a take() returned the wrong chunk", lines)
            check.that(have[3] == want[3], "a stream was in the wrong state after a call", lines)
    if final is not None:
        check.that(run.final == final, "a stream ended in the wrong state",
                   f"expected {final!r}\n" + detail)  # fmt: skip
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


def solve(ctx, requests, clients, **options):
    run = simulate(ctx, ctx.solution, requests, clients, **options)
    check_rules(run)
    return run


def matches_oracle(ctx, requests, clients, **options):
    got = solve(ctx, requests, clients, **options)
    want = simulate(ctx, ctx.oracle, requests, clients, **options)
    expect(got, steps=want.steps, releases=want.releases, takes=want.takes, final=want.final)
    return got


def req(id, arrival, prompt, output):
    return SimRequest(id, arrival, prompt, output)


def every(start, period, count):
    return Client([start + period * k for k in range(count)])


@case("the worked example: a slow client is paused, saved twice, then cut off", public=True)
def worked_example(ctx):
    requests = [req("a", 0, 8, 5), req("b", 0, 8, 8)]
    clients = {"a": every(15.5, 10, 5), "b": Client([25.5, 80.5])}
    run = solve(ctx, requests, clients, max_running=2, max_buffered=2, stall_ms=30)
    expect(
        run,
        steps=[(0, (("a", 8), ("b", 8)), ()), (12, (), ("a", "b")), (22, (), ("a",)),
               (31, (), ("a", "b")), (41, (), ("a", "b")), (80.5, (), ("b",)),
               (89.5, (), ("b",))],
        releases=[(128.5, "b")],
        takes=[(15.5, "a", [0], "open"), (25.5, "a", [1], "open"), (25.5, "b", [0, 1], "open"),
               (35.5, "a", [2], "open"), (45.5, "a", [3], "open"), (55.5, "a", [4], "finished"),
               (80.5, "b", [2, 3], "open")],
        final={"a": "finished", "b": "stalled"},
    )  # fmt: skip


@case("clients that keep up: plain continuous batching, one token per chunk", public=True)
def clients_keep_up(ctx):
    requests = [req(0, 0, 8, 4), req(1, 0, 16, 2), req(2, 5, 8, 3)]
    clients = {i: every(8.5, 4, 16) for i in range(3)}
    run = solve(ctx, requests, clients, max_running=2, max_buffered=4, stall_ms=50)
    expect(
        run,
        steps=[(0, ((0, 8), (1, 16)), ()), (14, (), (0, 1)), (24, ((2, 8),), (0,)),
               (35, (), (0, 2)), (45, (), (2,))],
        releases=[],
        final={0: "finished", 1: "finished", 2: "finished"},
    )  # fmt: skip
    check.that(run.peak_buffered() == 1, "the scenario lost its point")


@case("a client that never reads: generation stops at max_buffered tokens, then the "
      "stream is cut off and its slot reused", public=True)  # fmt: skip
def client_never_reads(ctx):
    # Rejects unbounded_buffer and stalled_not_released: "a" wants 50 tokens and its
    # client takes none; "b" waits for the only slot.
    requests = [req("a", 0, 8, 50), req("b", 1, 8, 2)]
    clients = {"b": every(61.5, 5, 4)}
    run = solve(ctx, requests, clients, max_running=1, max_buffered=3, stall_ms=20)
    expect(
        run,
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",)), (48, (("b", 8),), ()),
               (58, (), ("b",))],
        releases=[(48, "a")],
        takes=[(61.5, "b", [0], "open"), (66.5, "b", [], "open"), (71.5, "b", [1], "finished"),
               (76.5, "b", [], "finished")],
        final={"a": "stalled", "b": "finished"},
    )  # fmt: skip


@case("a slow but steady reader gets every token, in chunks of up to max_buffered",
      public=True)  # fmt: skip
def slow_reader_gets_everything(ctx):
    # Rejects drops_when_full, pauses_one_token_late, and take_does_not_wake: the
    # client reads every 40 ms, the buffer holds 3 tokens, the stall limit is 60 ms.
    run = solve(ctx, [req("a", 0, 8, 9)], {"a": every(40.5, 40, 4)}, max_running=1,
                max_buffered=3, stall_ms=60)  # fmt: skip
    expect(
        run,
        steps=[(0, (("a", 8),), ()), (10, (), ("a",)), (19, (), ("a",)), (40.5, (), ("a",)),
               (49.5, (), ("a",)), (58.5, (), ("a",)), (80.5, (), ("a",)), (89.5, (), ("a",)),
               (98.5, (), ("a",))],
        releases=[],
        takes=[(40.5, "a", [0, 1, 2], "open"), (80.5, "a", [3, 4, 5], "open"),
               (120.5, "a", [6, 7, 8], "finished"), (160.5, "a", [], "finished")],
        final={"a": "finished"},
    )  # fmt: skip


@case("close(): a running stream is released mid-step, a waiting one never starts, a "
      "finished one is unchanged", public=True)  # fmt: skip
def disconnects(ctx):
    # Rejects close_not_propagated. "a" is closed at t=26, during the step from 22 to
    # 32; "c" is closed at t=5 while waiting; "b" is closed at t=70 after finishing.
    requests = [req("a", 0, 8, 6), req("b", 0, 8, 6), req("c", 1, 8, 2), req("d", 2, 8, 2)]
    clients = {
        "a": Client([13.5], close_at=26),
        "b": Client([13.5, 23.5, 33.5, 43.5, 53.5, 63.5, 73.5], close_at=70),
        "c": Client(close_at=5),
        "d": every(42.5, 10, 4),
    }
    run = solve(ctx, requests, clients, max_running=2, max_buffered=4, stall_ms=100)
    expect(
        run,
        steps=[(0, (("a", 8), ("b", 8)), ()), (12, (), ("a", "b")), (22, (), ("a", "b")),
               (32, (("d", 8),), ("b",)), (43, (), ("b", "d")), (53, (), ("b",))],
        releases=[(26, "a")],
        final={"a": "closed", "b": "finished", "c": "closed", "d": "finished"},
    )  # fmt: skip


def stalling_scenario(stalls):
    """Six requests, all running at once; request 0's client reads every 6 ms, or
    (with `stalls`) stops reading between t=30 and t=400."""
    requests = [req(i, 0, 8, 24) for i in range(6)]
    clients = {i: reader(12.5, 6, 600) for i in range(6)}
    if stalls:
        clients[0] = reader(12.5, 6, 600, away=[(30, 400)])
    return requests, clients


@case("a client that stops reading does not delay any other stream", public=True)
def stalled_client_delays_nobody(ctx):
    # Rejects awaits_client_in_loop. With a slot for everyone, pausing one sequence
    # only makes steps shorter, so no token of another request may come later than
    # in the run where every client reads.
    options = {"max_running": 6, "max_buffered": 4, "stall_ms": 1000}
    stalled = solve(ctx, *stalling_scenario(True), **options)
    healthy = solve(ctx, *stalling_scenario(False), **options)
    for i in range(1, 6):
        got, base = stalled.token_times(i), healthy.token_times(i)
        check.that(len(got) == 24, "a stream with a reading client did not get all its tokens",
                   stalled.timeline())  # fmt: skip
        late = [k for k, (g, b) in enumerate(zip(got, base, strict=True)) if g > b]
        check.that(
            not late,
            "tokens of other requests were delayed while one client was not reading",
            f"request {i}: token {late[0] if late else 0} came later than in the run where "
            "every client reads\n" + stalled.timeline(),
        )
    check.that(len(stalled.token_times(0)) == 24, "the paused stream did not resume",
               stalled.timeline())  # fmt: skip
    matches_oracle(ctx, *stalling_scenario(True), **options)


@case("peak buffered tokens stay within max_buffered under slow and absent clients",
      public=True)  # fmt: skip
def peak_buffered_tokens(ctx):
    requests = workloads.poisson(ctx.rng, rate=0.04, n=40, prompt_tokens=(4, 64),
                                 output_tokens=(8, 60), quantum_s=1)  # fmt: skip
    clients = {}
    for r in requests:
        kind = r.id % 4
        if kind == 0:
            continue  # never reads
        period = (5, 45, 150)[kind - 1]
        clients[r.id] = reader(r.arrival_s + 0.5, period, r.arrival_s + 2500)
    run = simulate(ctx, ctx.solution, requests, clients, max_running=4, max_buffered=8,
                   stall_ms=200)  # fmt: skip
    check.at_most("peak_buffered_tokens", run.peak_buffered(), 8, unit="tokens")
    check_rules(run)


@case("ties: a take at the instant a step ends, and at the instant the stall is due")
def ties(ctx):
    # A take at the instant a token is due comes before it: the chunk is empty, and
    # with max_buffered = 1 the sequence pauses until the next take.
    run = solve(ctx, [req("a", 0, 8, 4)], {"a": Client([10, 19, 28, 37, 46])}, max_running=1,
                max_buffered=1, stall_ms=30)  # fmt: skip
    expect(
        run,
        steps=[(0, (("a", 8),), ()), (19, (), ("a",)), (37, (), ("a",))],
        releases=[(76, "a")],
        takes=[(10, "a", [], "open"), (19, "a", [0], "open"), (28, "a", [], "open"),
               (37, "a", [1], "open"), (46, "a", [], "open")],
        final={"a": "stalled"},
    )  # fmt: skip
    # The buffer fills at t=10; a take at exactly 10 + 30 comes first and saves the stream.
    run = solve(ctx, [req("a", 0, 8, 4)], {"a": Client([40, 99])}, max_running=1,
                max_buffered=1, stall_ms=30)  # fmt: skip
    expect(run, steps=[(0, (("a", 8),), ()), (40, (), ("a",))], releases=[(79, "a")],
           takes=[(40, "a", [0], "open"), (99, "a", [], "stalled")], final={"a": "stalled"})  # fmt: skip
    # A quarter of a millisecond later it is too late.
    run = solve(ctx, [req("a", 0, 8, 4)], {"a": Client([40.25, 99])}, max_running=1,
                max_buffered=1, stall_ms=30)  # fmt: skip
    expect(run, steps=[(0, (("a", 8),), ())], releases=[(40, "a")],
           takes=[(40.25, "a", [], "stalled"), (99, "a", [], "stalled")],
           final={"a": "stalled"})  # fmt: skip


@case("a stream whose last token fills the buffer is finished, never cut off")
def last_token_fills_the_buffer(ctx):
    # Rejects stall_timer_for_finished_streams: "a" has 2 tokens and max_buffered is 2.
    requests = [req("a", 0, 8, 2), req("b", 0, 8, 3)]
    clients = {"a": Client([200]), "b": Client([200])}
    run = solve(ctx, requests, clients, max_running=2, max_buffered=2, stall_ms=30)
    expect(
        run,
        steps=[(0, (("a", 8), ("b", 8)), ()), (12, (), ("a", "b"))],
        releases=[(52, "b")],
        takes=[(200, "a", [0, 1], "finished"), (200, "b", [], "stalled")],
        final={"a": "finished", "b": "stalled"},
    )


@case("a take long before the stall limit: the old timer must not cut the stream off later")
def stale_stall_timer(ctx):
    # Rejects stale_stall_timer: the buffer fills at t=28 (limit t=88); the client
    # takes at 40.5 and from then on every 15 ms, so the stream is healthy at 88.
    clients = {"a": Client([40.5, *(55.5 + 15 * k for k in range(12))])}
    run = solve(ctx, [req("a", 0, 8, 12)], clients, max_running=1, max_buffered=3, stall_ms=60)
    check.that(run.final == {"a": "finished"}, "a healthy stream was cut off", run.timeline())
    matches_oracle(ctx, [req("a", 0, 8, 12)], clients, max_running=1, max_buffered=3,
                   stall_ms=60)  # fmt: skip


def random_clients(rng, requests, *, span):
    clients = {}
    for r in requests:
        kind = int(rng.integers(0, 5))
        start = r.arrival_s + 0.125 + int(rng.integers(0, 20))
        if kind == 0:
            continue  # never reads
        if kind == 1:
            clients[r.id] = reader(start, int(rng.integers(2, 12)), start + span)
        elif kind == 2:
            clients[r.id] = reader(start, int(rng.integers(20, 90)), start + span)
        elif kind == 3:
            gone = start + int(rng.integers(10, 200))
            away = [(gone, gone + int(rng.integers(20, 400)))]
            clients[r.id] = reader(start, int(rng.integers(2, 12)), start + span, away=away)
        else:
            clients[r.id] = reader(start, int(rng.integers(2, 30)), start + span)
            clients[r.id].close_at = r.arrival_s + int(rng.integers(0, 300))
    return clients


@case("random traffic with fast, slow, stalling, and disconnecting clients")
def random_workloads(ctx):
    for max_running, rate, max_buffered, stall_ms in (
        (1, 0.01, 2, 40), (3, 0.03, 4, 80), (6, 0.05, 1, 25), (4, 0.05, 6, 150),
    ):  # fmt: skip
        requests = workloads.poisson(ctx.rng, rate=rate, n=50, prompt_tokens=(4, 96),
                                     output_tokens=(1, 24), quantum_s=1)  # fmt: skip
        clients = random_clients(ctx.rng, requests, span=700)
        matches_oracle(ctx, requests, clients, max_running=max_running,
                       max_buffered=max_buffered, stall_ms=stall_ms)  # fmt: skip


@case("bursts with clients on whole milliseconds: takes, stalls, and step ends coincide")
def bursty_workloads(ctx):
    for max_running, max_buffered in ((2, 1), (4, 3)):
        requests = workloads.bursty(ctx.rng, bursts=5, burst_size=6, period_s=160,
                                    prompt_tokens=8, output_tokens=(1, 10), quantum_s=1)  # fmt: skip
        clients = {}
        for r in requests:
            period = int(ctx.rng.integers(1, 40))
            start = r.arrival_s + int(ctx.rng.integers(0, 30))
            clients[r.id] = reader(start, period, start + int(ctx.rng.integers(50, 900)))
        matches_oracle(ctx, requests, clients, max_running=max_running,
                       max_buffered=max_buffered, stall_ms=30)  # fmt: skip
