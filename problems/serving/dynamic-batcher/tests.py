"""Tests for dynamic-batcher.

`simulate` is the simulator the statement describes: it owns virtual time (a SimClock
whose unit is the millisecond), the arrivals, the single timer, and the model, and it
drives the solution's DynamicBatcher one event at a time. Everything is recorded in
the clock's trace; the cases assert on the `dispatch` events.

Hand-checked scenarios are compared with literal schedules. The random workloads are
checked twice: against invariants computed here from the statement, and against the
oracle's trace.
"""

from learnai.judge import CheckFailure, case, check

DONE, ARRIVAL, TIMER = 0, 1, 2  # delivery order of events that share an instant
MAX_EVENTS = 100_000


class Run:
    """The outcome of one simulation."""

    def __init__(self, sim, arrivals, options):
        self.sim = sim
        self.arrivals = arrivals
        self.options = options

    @property
    def batches(self):
        """[(dispatch time in ms, [request ids])] in dispatch order."""
        return [(int(e.time), list(e["ids"])) for e in self.sim.trace.of("dispatch")]

    def timeline(self):
        """A readable log of the whole simulation, for failure details."""
        o = self.options
        settings = f"max_batch_size={o['max_batch_size']}, max_wait_ms={o['max_wait_ms']}"
        model = f"a batch of n requests runs for {o['base_ms']} + {o['per_item_ms']} * n ms"
        lines = [f"{settings}, {model}", "timeline:"]
        events = self.sim.trace.events
        for e in events[:60]:
            what = {
                "arrive": lambda e: f"on_arrival({e['id']!r})",
                "timer": lambda e: "on_timer",
                "done": lambda e: f"on_batch_done      (finished {e['ids']!r})",
                "dispatch": lambda e: f"  -> dispatch {e['ids']!r}",
            }[e.kind](e)
            lines.append(f"  t={int(e.time):>5}  {what}")
        if len(events) > 60:
            lines.append(f"  ... {len(events) - 60} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, batcher_class, arrivals, *, max_batch_size, max_wait_ms, base_ms, per_item_ms):
    """Drive a batcher with `arrivals`, a list of (time_ms, request_id) sorted by time.

    Events at the same instant are delivered in the order: batch completion, arrivals
    (in list order), timer. A dispatched batch of n requests keeps the model busy for
    base_ms + per_item_ms * n milliseconds.
    """
    sim = ctx.simclock()  # virtual time; this problem counts it in milliseconds
    options = {
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
        "base_ms": base_ms,
        "per_item_ms": per_item_ms,
    }
    run = Run(sim, arrivals, options)
    batcher = batcher_class(max_batch_size, max_wait_ms)
    waiting = []  # ids that arrived and are not dispatched yet, oldest first

    def fail(message, extra=""):
        raise CheckFailure(message, extra + run.timeline())

    async def drive():
        next_arrival, running, done_at, timer_at = 0, None, None, None
        for _ in range(MAX_EVENTS):
            due = []
            if done_at is not None:
                due.append((done_at, DONE))
            if next_arrival < len(arrivals):
                due.append((arrivals[next_arrival][0], ARRIVAL))
            if timer_at is not None:
                due.append((timer_at, TIMER))
            if not due:
                break
            when, kind = min(due)
            if when > sim.now():
                await sim.sleep(when - sim.now())
            now = int(when)

            if kind == DONE:
                sim.trace.record("done", ids=running)
                running, done_at = None, None
                batch, handler = batcher.on_batch_done(now), "on_batch_done"
            elif kind == ARRIVAL:
                request_id = arrivals[next_arrival][1]
                next_arrival += 1
                waiting.append(request_id)
                sim.trace.record("arrive", id=request_id)
                batch, handler = batcher.on_arrival(now, request_id), "on_arrival"
            else:
                timer_at = None
                sim.trace.record("timer")
                batch, handler = batcher.on_timer(now), "on_timer"

            if batch is not None and not isinstance(batch, list):
                fail(
                    f"{handler} must return a list of request ids or None",
                    f"it returned {batch!r}\n",
                )
            if batch:
                where = f"{handler} at t={now} returned {batch!r}\n"
                if running is not None:
                    fail("a batch was dispatched while the model was still busy", where)
                if len(batch) > max_batch_size:
                    fail("a batch is larger than max_batch_size", where)
                if len(set(batch)) != len(batch) or any(r not in waiting for r in batch):
                    fail(
                        "a batch contains a request that is not waiting "
                        "(unknown, already dispatched, or repeated)",
                        where,
                    )
                for request_id in batch:
                    waiting.remove(request_id)
                sim.trace.record("dispatch", ids=list(batch), size=len(batch))
                running = list(batch)
                done_at = now + base_ms + per_item_ms * len(batch)

            wake = batcher.next_timer_ms()
            if wake is not None and (isinstance(wake, bool) or not isinstance(wake, int)):
                fail("next_timer_ms must return an int or None", f"it returned {wake!r}\n")
            timer_at = wake if wake is not None and wake > now else None
        else:
            fail(
                "the simulation did not come to rest: on_timer keeps being requested "
                "without anything being dispatched"
            )

    sim.run_until_complete(drive())
    if waiting:
        fail(
            f"{len(waiting)} of {len(arrivals)} requests were never dispatched",
            f"still waiting when the simulation went idle: {waiting[:10]!r}\n",
        )
    return run


def numbered(times):
    """Arrivals with request ids 0, 1, 2, ... in arrival order."""
    return [(t, i) for i, t in enumerate(times)]


def expect(run, expected):
    """Compare a run's dispatches with a schedule."""
    got = run.batches
    detail = (
        f"expected dispatches (time, ids): {expected!r}\n"
        f"got dispatches      (time, ids): {got!r}\n" + run.timeline()
    )
    for k, ((got_time, got_ids), (want_time, want_ids)) in enumerate(
        zip(got, expected, strict=False), start=1
    ):
        check.that(
            got_ids == want_ids, f"batch {k} holds {got_ids!r}, expected {want_ids!r}", detail
        )
        check.that(
            got_time == want_time,
            f"batch {k} was dispatched at t={got_time!r}, expected t={want_time!r}",
            detail,
        )
    check.that(
        len(got) == len(expected),
        f"{len(got)} batches were dispatched, expected {len(expected)}",
        detail,
    )


def solve(ctx, arrivals, **options):
    return simulate(ctx, ctx.solution.DynamicBatcher, arrivals, **options)


def check_invariants(run):
    """The policy, restated as properties of the trace (independent of the oracle)."""
    size_limit, max_wait = run.options["max_batch_size"], run.options["max_wait_ms"]
    arrival_time = {request_id: t for t, request_id in run.arrivals}
    order = [request_id for _, request_id in run.arrivals]
    batches = run.batches
    detail = run.timeline()

    dispatched = [request_id for _, ids in batches for request_id in ids]
    check.that(
        sorted(map(repr, dispatched)) == sorted(map(repr, order)),
        "some request was dropped or dispatched twice",
        detail,
    )
    check.that(dispatched == order, "requests were not served first-come first-served", detail)

    done_times = [int(e.time) for e in run.sim.trace.of("done")]
    start = 0  # index in `order` of the oldest request of the current batch
    for k, (when, ids) in enumerate(batches):
        free_at = done_times[k - 1] if k else 0  # when the model became idle
        deadline = arrival_time[ids[0]] + max_wait
        check.that(1 <= len(ids) <= size_limit, "a batch is empty or over max_batch_size", detail)
        check.that(
            when <= max(free_at, deadline),
            "a request waited past its deadline although the model was idle",
            f"batch {k + 1} {ids!r}: dispatched at {when}, its oldest request's deadline "
            f"was {deadline}, the model was idle from {free_at}\n" + detail,
        )
        if start + size_limit <= len(order):
            full_at = arrival_time[order[start + size_limit - 1]]
            check.that(
                when <= max(free_at, full_at),
                "a full batch was waiting at an idle model and was not dispatched at once",
                f"batch {k + 1} {ids!r}: dispatched at {when}; max_batch_size requests were "
                f"waiting from {full_at}, the model was idle from {free_at}\n" + detail,
            )
        if len(ids) < size_limit:
            check.that(
                when >= deadline,
                "a partial batch was dispatched before its oldest request's deadline",
                f"batch {k + 1} {ids!r}: dispatched at {when}, deadline {deadline}\n" + detail,
            )
            waiting_before = sum(1 for r in order[start:] if arrival_time[r] < when)
            check.that(
                len(ids) >= min(size_limit, waiting_before),
                "a batch left out requests that were waiting although it had room",
                f"batch {k + 1} {ids!r}: {waiting_before} requests were waiting\n" + detail,
            )
        start += len(ids)


def random_workload(rng, n):
    """A bursty arrival pattern with shuffled request ids, plus random settings."""
    times, t = [], 0
    for _ in range(n):
        roll = rng.random()
        if roll < 0.35:
            gap = 0  # simultaneous arrivals
        elif roll < 0.85:
            gap = int(rng.integers(1, 8))
        else:
            gap = int(rng.integers(20, 90))  # a lull, long enough to drain the queue
        t += gap
        times.append(t)
    ids = [int(i) for i in rng.permutation(n)]
    options = {
        "max_batch_size": int(rng.integers(1, 9)),
        "max_wait_ms": int(rng.integers(0, 31)),
        "base_ms": int(rng.integers(4, 21)),
        "per_item_ms": int(rng.integers(0, 4)),
    }
    return list(zip(times, ids, strict=True)), options


# The model in the worked example: a batch of n requests runs for 16 + 2 * n ms.
EXAMPLE_MODEL = {"base_ms": 16, "per_item_ms": 2}


@case("the worked example: timeout, backlog, late flush, and a batch that fills", public=True)
def worked_example(ctx):
    arrivals = numbered([0, 4, 25, 26, 27, 28, 29, 80, 84, 86])
    run = solve(ctx, arrivals, max_batch_size=3, max_wait_ms=10, **EXAMPLE_MODEL)
    expect(run, [(10, [0, 1]), (30, [2, 3, 4]), (52, [5, 6]), (86, [7, 8, 9])])


@case("a lone request is dispatched when it has waited max_wait_ms", public=True)
def lone_request(ctx):
    run = solve(ctx, numbered([7]), max_batch_size=4, max_wait_ms=50, base_ms=10, per_item_ms=1)
    expect(run, [(57, [0])])


@case("a batch is dispatched the moment it reaches max_batch_size", public=True)
def fills_up(ctx):
    # Rejects full_batch_off_by_one.
    options = {"max_batch_size": 4, "max_wait_ms": 50, "base_ms": 10, "per_item_ms": 1}
    expect(solve(ctx, numbered([0, 1, 2, 3]), **options), [(3, [0, 1, 2, 3])])
    expect(solve(ctx, numbered([5, 5, 5, 5]), **options), [(5, [0, 1, 2, 3])])
    # The fifth request finds the model busy; at t=17 its deadline (55) is still ahead.
    expect(solve(ctx, numbered([0, 1, 2, 3, 3]), **options), [(3, [0, 1, 2, 3]), (53, [4])])


@case("the wait is measured from the oldest request, not the newest", public=True)
def oldest_sets_the_deadline(ctx):
    # Rejects waits_from_newest.
    options = {"max_batch_size": 8, "max_wait_ms": 10, "base_ms": 5, "per_item_ms": 0}
    expect(solve(ctx, numbered([0, 6]), **options), [(10, [0, 1])])
    run = solve(ctx, numbered([0, 3, 6, 9, 12, 15]), **options)
    expect(run, [(10, [0, 1, 2, 3]), (22, [4, 5])])


@case("requests that arrive while the model is busy wait for it to finish", public=True)
def arrivals_while_busy(ctx):
    # Rejects dispatches_while_busy: a full batch forms at t=4 and a deadline passes at
    # t=12, both while the first batch (t=1 to t=21) is still running.
    options = {"max_batch_size": 2, "max_wait_ms": 10, "base_ms": 20, "per_item_ms": 0}
    run = solve(ctx, numbered([0, 1, 2, 4, 6]), **options)
    expect(run, [(1, [0, 1]), (21, [2, 3]), (41, [4])])


@case("an arrival exactly at the oldest request's deadline joins that batch")
def arrival_at_the_deadline(ctx):
    options = {"max_batch_size": 4, "max_wait_ms": 10, "base_ms": 5, "per_item_ms": 0}
    expect(solve(ctx, numbered([0, 10]), **options), [(10, [0, 1])])
    # Two arrivals at the deadline: the first triggers the dispatch, the second is late.
    expect(solve(ctx, numbered([0, 10, 10]), **options), [(10, [0, 1]), (20, [2])])
    # One millisecond earlier and one later, for contrast.
    expect(solve(ctx, numbered([0, 9]), **options), [(10, [0, 1])])
    expect(solve(ctx, numbered([0, 11]), **options), [(10, [0]), (21, [1])])


@case("batch completion: an expired deadline, a pending one, and a simultaneous arrival")
def at_completion(ctx):
    options = {"max_batch_size": 2, "max_wait_ms": 10, "base_ms": 20, "per_item_ms": 0}
    # Request 2 (deadline 15) expired while the model was busy: it leaves at t=21, in
    # on_batch_done, before the arrival at t=21 is delivered.
    run = solve(ctx, numbered([0, 1, 5, 21]), **options)
    expect(run, [(1, [0, 1]), (21, [2]), (41, [3])])
    # Request 2 (deadline 25) has not expired at t=21; the arrival at t=21 fills the batch.
    run = solve(ctx, numbered([0, 1, 15, 21]), **options)
    expect(run, [(1, [0, 1]), (21, [2, 3])])
    # Not expired and nothing else arrives: it waits for its own deadline, not longer.
    run = solve(ctx, numbered([0, 1, 15]), **options)
    expect(run, [(1, [0, 1]), (25, [2])])
    # The deadline falls exactly on the completion time.
    run = solve(ctx, numbered([0, 1, 11]), **options)
    expect(run, [(1, [0, 1]), (21, [2])])


@case("requests left over after a full batch keep their own deadline")
def leftovers_keep_their_deadline(ctx):
    # Rejects timer_not_rearmed_after_flush, stale_deadline_after_flush, and
    # wait_restarts_at_completion: request 4 arrived at t=4, so its deadline is 34.
    options = {"max_batch_size": 2, "max_wait_ms": 30, "base_ms": 10, "per_item_ms": 0}
    run = solve(ctx, numbered([0, 1, 2, 3, 4]), **options)
    expect(run, [(1, [0, 1]), (11, [2, 3]), (34, [4])])
    run = solve(ctx, numbered([0, 1, 2, 3, 4, 5, 6, 60]), **options)
    expect(run, [(1, [0, 1]), (11, [2, 3]), (21, [4, 5]), (36, [6]), (90, [7])])


@case("max_wait_ms = 0 and max_batch_size = 1")
def degenerate_settings(ctx):
    options = {"max_batch_size": 3, "max_wait_ms": 0, "base_ms": 10, "per_item_ms": 0}
    run = solve(ctx, numbered([0, 0, 0, 4, 10, 10]), **options)
    expect(run, [(0, [0]), (10, [1, 2, 3]), (20, [4, 5])])
    options = {"max_batch_size": 1, "max_wait_ms": 100, "base_ms": 5, "per_item_ms": 0}
    run = solve(ctx, numbered([0, 1, 2, 40]), **options)
    expect(run, [(0, [0]), (5, [1]), (10, [2]), (40, [3])])


@case("a burst larger than max_batch_size, with opaque request ids")
def big_burst(ctx):
    # Rejects lifo_batch.
    names = ["req-k", "req-c", "req-x", "req-a", "req-m", "req-b", "req-z"]
    arrivals = [(100, name) for name in names]
    run = solve(ctx, arrivals, max_batch_size=3, max_wait_ms=25, base_ms=8, per_item_ms=4)
    expect(run, [(100, names[0:3]), (120, names[3:6]), (140, names[6:7])])


@case("random bursty workloads: no drops, FIFO, size limit, and no avoidable waiting")
def random_invariants(ctx):
    for _ in range(12):
        arrivals, options = random_workload(ctx.rng, 150)
        check_invariants(solve(ctx, arrivals, **options))


@case("random bursty workloads: the same batches at the same times as the reference")
def random_matches_reference(ctx):
    for _ in range(12):
        arrivals, options = random_workload(ctx.rng, 150)
        got = solve(ctx, arrivals, **options)
        want = simulate(ctx, ctx.oracle.DynamicBatcher, arrivals, **options)
        expect(got, want.batches)
