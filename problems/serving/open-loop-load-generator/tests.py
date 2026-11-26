"""Tests for open-loop-load-generator.

`play` runs `generate_load` on a simulated clock against a scripted server: request
`i` takes `durations[i]` seconds and then succeeds or fails. The test records the
virtual time of every `send(i)` call. What the statement defines is computed here in
closed form (`plan`): the intended times from the uniforms, and, with a concurrency
limit, the send times of a first-come-first-served queue. Times are compared to 1e-9 s.
"""

import heapq
import math

import numpy as np

from learnai.judge import CheckFailure, case, check

TOL = 1e-9
KEYS = ("intended_s", "sent_s", "done_s", "latency_s", "ok")


def gaps_to_uniforms(gaps, rate):
    """Uniforms that make the gaps come out as the given numbers of seconds."""
    return [1.0 - math.exp(-gap * rate) for gap in gaps]


class Run:
    def __init__(self):
        self.calls = []  # (virtual time, i) of each send call
        self.in_flight = 0
        self.peak_in_flight = 0
        self.records = None
        self.error = None
        self.returned_at = None


def play(ctx, module, *, rate, uniforms, durations, fail=(), start=0.0, max_in_flight=None):
    n = len(durations)
    sim = ctx.simclock()
    run = Run()

    async def send(i):
        run.calls.append((sim.now(), i))
        if len(run.calls) > n + 50:
            raise CheckFailure("generate_load sent far more requests than n")
        run.in_flight += 1
        run.peak_in_flight = max(run.peak_in_flight, run.in_flight)
        await sim.sleep(durations[i])
        run.in_flight -= 1
        if i in fail:
            raise ConnectionError(f"request {i}: connection reset by peer")
        return {"status": 200}

    async def main():
        try:
            if max_in_flight is None:
                run.records = await module.generate_load(send, rate, n, list(uniforms))
            else:
                run.records = await module.generate_load(
                    send, rate, n, list(uniforms), max_in_flight=max_in_flight
                )
        except OSError as error:  # a scripted failure got out; `verify` reports it
            run.error = error
        run.returned_at = sim.now()

    sim.spawn_at(start, main)
    sim.run()
    return run


def plan(*, rate, uniforms, durations, start=0.0, max_in_flight=None):
    """What the statement specifies: (intended, sent, done) per request."""
    intended, t = [], start
    for u in uniforms[: len(durations)]:
        t += -math.log(1.0 - u) / rate
        intended.append(t)
    sent, done, free = [], [], []
    for i, due in enumerate(intended):
        when = due
        if max_in_flight is not None and len(free) == max_in_flight:
            when = max(due, heapq.heappop(free))  # the earliest slot, in request order
        sent.append(when)
        done.append(when + durations[i])
        heapq.heappush(free, done[i])
    return intended, sent, done


def describe(options):
    durations, uniforms = options["durations"], options["uniforms"]
    more_u = " ..." if len(uniforms) > 12 else ""
    more_d = " ..." if len(durations) > 12 else ""
    header = f"rate = {options['rate']!r}, n = {len(durations)}, "
    header += f"max_in_flight = {options.get('max_in_flight')!r}; "
    header += f"generate_load is called at t = {options.get('start', 0.0)!r}"
    lines = [
        header,
        f"uniforms = {[round(float(u), 6) for u in uniforms[:12]]!r}{more_u}",
        f"request i takes (s): {[float(d) for d in durations[:12]]!r}{more_d}",
    ]
    if options.get("fail"):
        lines.append(f"requests that fail: {sorted(options['fail'])!r}")
    return "\n".join(lines) + "\n"


def close(a, b):
    return isinstance(a, (int, float)) and not isinstance(a, bool) and abs(a - b) <= TOL


def verify(run, options):
    """Check a run against `plan` (independent of the oracle)."""
    n = len(options["durations"])
    intended, sent, done = plan(**{k: v for k, v in options.items() if k != "fail"})
    fail = set(options.get("fail", ()))
    table = "expected (i, intended_s, sent_s, done_s, latency_s, ok):\n" + "".join(
        f"  {i}: {intended[i]:.6f}  {sent[i]:.6f}  {done[i]:.6f}  "
        f"{done[i] - intended[i]:.6f}  {i not in fail}\n"
        for i in range(min(n, 16))
    )
    detail = describe(options) + table
    detail += (
        f"send(i) was called at (time, i): {[(round(t, 6), i) for t, i in run.calls[:16]]!r}\n"
    )
    check.that(
        run.error is None,
        "generate_load raised instead of returning; a failed request must be recorded "
        "with ok = False",
        f"{type(run.error).__name__}: {run.error}\n" + detail,
    )
    check.that(
        sorted(i for _, i in run.calls) == list(range(n)),
        f"send was called {len(run.calls)} times, expected {n} (once per request)",
        detail,
    )
    called = {i: when for when, i in run.calls}
    limit = options.get("max_in_flight")
    if limit is None:
        late = [i for i in range(n) if not close(called[i], intended[i])]
        check.that(
            not late,
            "a request was not sent at its intended time",
            f"first at request {late[0] if late else 0}\n" + detail,
        )
    else:
        check.that(
            run.peak_in_flight <= limit,
            f"{run.peak_in_flight} requests were in flight at once, more than max_in_flight",
            detail,
        )
        late = [i for i in range(n) if not close(called[i], sent[i])]
        check.that(
            not late,
            "with max_in_flight, a request was not sent at the first instant at or after "
            "its intended time at which it was first in line for a free slot",
            f"first at request {late[0] if late else 0}\n" + detail,
        )
    records = run.records
    check.that(
        isinstance(records, list) and len(records) == n,
        "generate_load must return a list with one record per request",
        f"it returned {str(records)[:200]}\n" + detail,
    )
    for i, record in enumerate(records):
        got = f"record {i}: {record!r}\n"
        check.that(
            isinstance(record, dict) and set(record) == set(KEYS),
            "a record is not a dict with the five specified keys",
            got + detail,
        )
        check.that(
            close(record["intended_s"], intended[i]),
            f"record {i}: intended_s is {record['intended_s']!r}, expected {intended[i]!r} (the "
            "arrival time of the Poisson schedule)",
            got + detail,
        )
        check.that(
            close(record["sent_s"], called[i]),
            f"record {i}: sent_s is {record['sent_s']!r}, expected {called[i]!r} (the time at "
            "which send was called)",
            got + detail,
        )
        check.that(
            close(record["done_s"], done[i]),
            f"record {i}: done_s is {record['done_s']!r}, expected {done[i]!r} (the time at "
            "which the request ended)",
            got + detail,
        )
        check.that(
            close(record["latency_s"], done[i] - intended[i]),
            f"record {i}: latency_s is {record['latency_s']!r}, expected "
            f"{done[i] - intended[i]!r} (measured from the intended send time)",
            got + detail,
        )
        check.that(
            record["ok"] is (i not in fail),
            f"record {i}: ok is {record['ok']!r}, expected {i not in fail}",
            got + detail,
        )
    end = max([options.get("start", 0.0), *done])
    check.that(
        close(run.returned_at, end),
        "generate_load did not return at the instant the last request ended",
        f"it returned at t={run.returned_at!r}; the last request ended at t={end!r}\n" + detail,
    )


def solve(ctx, **options):
    run = play(ctx, ctx.solution, **options)
    verify(run, options)
    return run


EXAMPLE = {
    "rate": 2.0,
    "uniforms": gaps_to_uniforms([0.5, 0.25, 0.25, 1.0], 2.0),
    "durations": [1.0, 0.25, 0.25, 0.25],
}


@case("the worked example: four requests, the first one slow", public=True)
def worked_example(ctx):
    run = solve(ctx, **EXAMPLE)
    check.allclose([t for t, _ in run.calls], [0.5, 0.75, 1.0, 2.0], label="send times")
    check.allclose([r["latency_s"] for r in run.records], [1.0, 0.25, 0.25, 0.25],
                   label="latency_s")  # fmt: skip


@case("the worked example with max_in_flight = 1: late sends, latency from the intended "
      "time", public=True)  # fmt: skip
def worked_example_limited(ctx):
    # Rejects latency_from_send_time and ignores_the_limit.
    run = solve(ctx, **EXAMPLE, max_in_flight=1)
    check.allclose([t for t, _ in run.calls], [0.5, 1.5, 1.75, 2.0], label="send times")
    check.allclose([r["latency_s"] for r in run.records], [1.0, 1.0, 1.0, 0.25],
                   label="latency_s")  # fmt: skip


@case("a server slower than the arrival rate does not slow the arrivals down", public=True)
def slow_server(ctx):
    # Rejects closed_loop and does_not_wait_for_responses: every response takes two
    # seconds and the requests are a quarter of a second apart.
    solve(ctx, rate=4.0, uniforms=gaps_to_uniforms([0.25] * 12, 4.0), durations=[2.0] * 12)


@case("a failed request is recorded with ok = False and the run goes on", public=True)
def failed_requests(ctx):
    # Rejects first_failure_aborts.
    solve(ctx, rate=8.0, uniforms=gaps_to_uniforms([0.125] * 8, 8.0),
          durations=[0.5, 0.25, 1.0, 0.125, 0.5, 0.25, 0.125, 0.5], fail={1, 2, 6})  # fmt: skip
    solve(ctx, rate=8.0, uniforms=gaps_to_uniforms([0.125] * 4, 8.0),
          durations=[0.5, 0.25, 1.0, 0.125], fail={0, 3}, max_in_flight=2)  # fmt: skip


@case("the gaps come from the uniforms: -ln(1 - u) / rate", public=True)
def gaps_from_uniforms(ctx):
    # Rejects uniform_gaps. u = 0 gives a gap of zero: two requests at one instant.
    uniforms = [0.5, 0.0, 0.9, 0.25, 0.75, 0.0, 0.99, 0.1]
    run = solve(ctx, rate=1.0, uniforms=uniforms, durations=[0.5] * 8)
    check.allclose(run.calls[0][0], math.log(2.0), label="the first send time")
    solve(ctx, rate=50.0, uniforms=uniforms, durations=[0.5] * 8)


@case("generate_load called at t = 3.5: the schedule starts then")
def starts_later(ctx):
    # Rejects ignores_start_time.
    solve(ctx, **EXAMPLE, start=3.5)
    solve(ctx, **EXAMPLE, start=3.5, max_in_flight=2)


@case("no requests, and a single request")
def degenerate(ctx):
    run = solve(ctx, rate=5.0, uniforms=[], durations=[], start=1.0)
    check.equal(run.records, [], label="generate_load with n = 0")
    solve(ctx, rate=5.0, uniforms=[0.3], durations=[0.75])
    # More uniforms than requests: the extra ones are not used.
    solve(ctx, rate=5.0, uniforms=[0.3, 0.6, 0.9], durations=[0.75, 0.5])


@case("max_in_flight: waiting requests go out first come, first served, the schedule "
      "does not shift")  # fmt: skip
def limited_queue(ctx):
    # Rejects relative_sleeps_slip: a long stall, then requests that are due while
    # others wait, then a quiet stretch in which the queue catches up.
    gaps = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 4.0, 0.25, 0.25, 0.25]
    durations = [3.0, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 1.0, 0.25, 0.25]
    for limit in (1, 2, 3, 10):
        solve(ctx, rate=4.0, uniforms=gaps_to_uniforms(gaps, 4.0), durations=durations,
              max_in_flight=limit)  # fmt: skip


@case("random schedules and service times, with and without a limit")
def random_runs(ctx):
    for rate, limit in ((2.0, None), (20.0, None), (20.0, 3), (5.0, 1), (100.0, 8)):
        n = 60
        uniforms = ctx.rng.random(n).tolist()
        durations = (ctx.rng.exponential(2.0 / rate, n) + 0.01).tolist()
        fail = {int(i) for i in ctx.rng.choice(n, 6, replace=False)}
        solve(ctx, rate=rate, uniforms=uniforms, durations=durations, fail=fail,
              max_in_flight=limit, start=float(ctx.rng.integers(0, 5)))  # fmt: skip


@case("coordinated omission: one slow response must show in the latencies of the "
      "requests that were due behind it")  # fmt: skip
def coordinated_omission(ctx):
    # One connection; request 0 stalls for 5 s while 19 more requests fall due.
    # Measured from the send time every later latency would be 0.01 s.
    n = 20
    options = {"rate": 10.0, "uniforms": gaps_to_uniforms([0.1] * n, 10.0),
               "durations": [5.0] + [0.01] * (n - 1), "max_in_flight": 1}  # fmt: skip
    run = solve(ctx, **options)
    slow = sum(1 for r in run.records if r["latency_s"] > 1.0)
    check.that(
        slow == n,
        "requests that were due during a stall report a latency that leaves the stall out",
        f"{slow} of {n} latencies are above 1 s; all {n} requests were due before the stalled "
        "response came back\n" + describe(options),
    )


@case("statistical: with a slow server the gaps between sends are still exponential")
def gaps_are_exponential(ctx):
    # For exponential gaps with mean 1 / rate, exp(-rate * gap) is uniform on (0, 1]
    # and has mean 1/2. Uniform gaps with the same mean give 0.43, constant gaps 0.37,
    # and a closed loop, whose gaps cannot be shorter than a response, far less.
    n, rate = 20_000, 50.0
    uniforms = ctx.rng.random(n).tolist()
    durations = np.full(n, 4.0 / rate).tolist()
    run = play(ctx, ctx.solution, rate=rate, uniforms=uniforms, durations=durations)
    check.that(run.error is None, "generate_load raised", f"{run.error!r}\n")
    check.that(len(run.calls) == n, "send was not called once per request")
    times = np.sort(np.array([t for t, _ in run.calls]))
    gaps = np.diff(np.concatenate([[0.0], times]))
    check.mean(np.exp(-rate * gaps), 0.5, low=0.0, high=1.0, label="mean of exp(-rate * gap)")
    ctx.metric("mean_gap_ms", round(float(gaps.mean() * 1000), 3), "ms")
