"""Tests for stream-latency-probe.

A scripted server plays each request on a simulated clock: it accepts the connection
after `connect` seconds, delivers each chunk at a fixed offset from the moment the
request was sent, and closes (or breaks) the stream at `close`. Chunks are delivered at
absolute virtual times whether or not the client is ready, as a real server's are; a
client that asks late gets the chunk at once.

Every time in the scripts is a multiple of 1/1024 s, so every timestamp and every
difference is exact in binary; the 1e-9 tolerance is only a courtesy. Hand-checked
scripts are compared with literal records, everything else with the oracle run through
the same simulation.
"""

import numpy as np

from whetstone.judge import CheckFailure, case, check

TOL = 1e-9
KEYS = ("status", "t_start", "token_times", "ttft", "itl", "tpot", "e2e")
SUMMARY_KEYS = ("requests", "ok", "ttft", "tpot", "itl", "e2e")


def script(connect, chunks, close, fail=None):
    """One request: `chunks` is a list of (offset_s, token_count); `fail` is None,
    "open" (refused after `connect`), or "stream" (broken at `close`)."""
    return {"connect": connect, "chunks": chunks, "close": close, "fail": fail}


def describe(s):
    lines = [f"  +{s['connect']:g} s: " + ("refused" if s["fail"] == "open" else "response begins")]
    if s["fail"] != "open":
        for offset, count in s["chunks"]:
            lines.append(f"  +{offset:g} s: a chunk of {count} token{'s' * (count != 1)}")
        lines.append(
            f"  +{s['close']:g} s: " + ("the stream breaks" if s["fail"] == "stream" else "end")
        )
    return "\n".join(lines) + "\n"


async def until(sim, when):
    if when > sim.now():
        reached = sim.loop.create_future()
        sim.call_at(when, reached.set_result, None)
        await reached


class Server:
    """Plays scripts and records what the client did."""

    def __init__(self, sim, scripts, sink_s=0.0):
        self.sim, self.scripts, self.sink_s = sim, scripts, sink_s
        self.opened = {}  # request index -> time open was called
        self.delivered = []  # token-id lists, in delivery order
        self.sunk = []  # what the sink received, in order
        self.in_sink = 0
        self.problems = []

    def open(self, index):
        async def open_stream():
            if index in self.opened:
                self.problems.append(f"request {index} was sent twice")
            self.opened[index] = self.sim.now()
            s = self.scripts[index]
            await until(self.sim, self.opened[index] + s["connect"])
            if s["fail"] == "open":
                raise ConnectionRefusedError("connection refused")
            return _Stream(self, index, self.opened[index])

        return open_stream

    async def sink(self, chunk):
        self.sunk.append(list(chunk))
        self.in_sink += 1
        if self.sink_s:
            await self.sim.sleep(self.sink_s)
        self.in_sink -= 1


class _Stream:
    def __init__(self, server, index, opened):
        self.server, self.index, self.opened = server, index, opened
        self.script = server.scripts[index]
        self.position = 0
        self.next_token = index * 1000

    def __aiter__(self):
        return self

    async def __anext__(self):
        server, s = self.server, self.script
        if server.in_sink and len(server.scripts) == 1:
            server.problems.append("the next chunk was requested before sink returned")
        if self.position < len(s["chunks"]):
            offset, count = s["chunks"][self.position]
            self.position += 1
            await until(server.sim, self.opened + offset)
            chunk = list(range(self.next_token, self.next_token + count))
            self.next_token += count
            server.delivered.append(chunk)
            return list(chunk)
        await until(server.sim, self.opened + s["close"])
        if s["fail"] == "stream":
            raise ConnectionResetError("connection reset by peer")
        raise StopAsyncIteration


def finish(sim, server, holder, what):
    sim.run()
    if "result" not in holder:
        raise CheckFailure(f"{what} never returned", "the simulation went idle first\n")
    if server.problems:
        raise CheckFailure(f"{what}: {server.problems[0]}", "\n".join(server.problems) + "\n")
    # With several requests in flight only the multiset is defined: two chunks that
    # arrive at one instant may reach the sink in either order.
    in_order = len(server.scripts) == 1
    same = (
        server.sunk == server.delivered
        if in_order
        else sorted(server.sunk) == sorted(server.delivered)
    )
    if not same:
        raise CheckFailure(
            f"{what}: the sink did not receive every chunk, once, in order",
            f"delivered: {server.delivered[:12]!r}\n     sunk: {server.sunk[:12]!r}\n",
        )
    return holder["result"]


def run_probe(ctx, module, s, *, start=0.0, sink_s=0.0):
    sim = ctx.simclock()
    server = Server(sim, [s], sink_s)
    holder = {}

    async def main():
        holder["result"] = await module.probe(server.open(0), server.sink)

    sim.spawn_at(start, main)
    return finish(sim, server, holder, "probe")


def run_loop(ctx, module, scripts, concurrency, *, start=0.0, sink_s=0.0):
    sim = ctx.simclock()
    server = Server(sim, scripts, sink_s)
    holder = {}

    async def main():
        def open_request(i):
            return server.open(i)()

        holder["result"] = await module.run_closed_loop(
            open_request, server.sink, concurrency, len(scripts)
        )

    sim.spawn_at(start, main)
    return finish(sim, server, holder, "run_closed_loop"), server.opened


def same_number(got, want):
    if want is None or got is None:
        return got is None and want is None
    return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) <= TOL


def same_list(got, want):
    return (
        isinstance(got, list)
        and len(got) == len(want)
        and all(same_number(g, w) for g, w in zip(got, want, strict=True))
    )


def check_record(got, want, label, context=""):
    detail = f"{context}expected: {want!r}\n     got: {got!r}\n"
    check.that(isinstance(got, dict), f"{label}: the record must be a dict", detail)
    check.that(
        set(got) == set(KEYS), f"{label}: the record does not have the specified keys", detail
    )
    check.that(got["status"] == want["status"], f"{label}: wrong status {got['status']!r}", detail)
    for key in ("t_start", "ttft", "tpot", "e2e"):
        check.that(same_number(got[key], want[key]), f"{label}: {key!r} is wrong", detail)
    for key in ("token_times", "itl"):
        check.that(same_list(got[key], want[key]), f"{label}: {key!r} is wrong", detail)


def check_probe(ctx, s, label, *, start=0.0, sink_s=0.0, want=None):
    if want is None:
        want = run_probe(ctx, ctx.oracle, s, start=start, sink_s=sink_s)
    got = run_probe(ctx, ctx.solution, s, start=start, sink_s=sink_s)
    context = f"request sent at t={start:g} s; the sink takes {sink_s:g} s per chunk\n" + describe(
        s
    )
    check_record(got, want, label, context)


def check_loop(ctx, scripts, concurrency, label, *, sink_s=0.0, want_starts=None):
    want, want_opened = run_loop(ctx, ctx.oracle, scripts, concurrency, sink_s=sink_s)
    got, opened = run_loop(ctx, ctx.solution, scripts, concurrency, sink_s=sink_s)
    if want_starts is not None:
        assert [want_opened[i] for i in range(len(scripts))] == want_starts
    starts = [opened.get(i) for i in range(len(scripts))]
    expected = [want_opened[i] for i in range(len(scripts))]
    check.that(
        None not in starts,
        f"{label}: a request was never sent",
        f"send times by request: {starts!r}\n",
    )
    check.that(
        all(abs(a - b) <= TOL for a, b in zip(starts, expected, strict=True)),
        f"{label}: a request was sent at the wrong time",
        f"concurrency = {concurrency}\nexpected send times: {expected!r}\n"
        f"     got send times: {starts!r}\n",
    )
    check.that(
        isinstance(got, list) and len(got) == len(want),
        f"{label}: the result must be a list with one record per request",
        f"got: {got!r}\n",
    )
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        check_record(g, w, f"{label}, record {i}")


def check_summary(ctx, results, q, label, want=None):
    got = ctx.call("summarize", [dict(r) for r in results], q)
    if want is None:
        want = ctx.expected("summarize", [dict(r) for r in results], q)
    detail = f"q = {q}\nexpected: {want!r}\n     got: {got!r}\n"
    check.that(isinstance(got, dict), f"{label}: the summary must be a dict", detail)
    check.that(
        set(got) == set(SUMMARY_KEYS),
        f"{label}: the summary does not have the specified keys",
        detail,
    )
    for key in ("requests", "ok"):
        check.that(
            type(got[key]) is int and got[key] == want[key], f"{label}: {key!r} is wrong", detail
        )
    for key in ("ttft", "tpot", "itl", "e2e"):
        check.that(same_number(got[key], want[key]), f"{label}: {key!r} is wrong", detail)


def record(status, t_start, token_times):
    """A record as the statement defines it, built from its timestamps."""
    r = {"status": status, "t_start": t_start, "token_times": list(token_times),
         "ttft": None, "itl": [], "tpot": None, "e2e": None}  # fmt: skip
    if status == "ok":
        n = len(token_times)
        r["ttft"] = token_times[0] - t_start
        r["itl"] = [token_times[i] - token_times[i - 1] for i in range(1, n)]
        r["tpot"] = (token_times[-1] - token_times[0]) / (n - 1) if n > 1 else None
        r["e2e"] = token_times[-1] - t_start
    return r


# The statement's example: a role-only chunk, three chunks of 2, 1, and 3 tokens, a
# usage chunk without tokens, then the end of the stream.
EXAMPLE = script(0.25, [(0.5, 0), (0.75, 2), (0.875, 1), (1.0, 3), (1.0625, 0)], 1.125)
EXAMPLE_RECORD = {
    "status": "ok",
    "t_start": 2.0,
    "token_times": [2.75, 2.75, 2.875, 3.0, 3.0, 3.0],
    "ttft": 0.75,
    "itl": [0.0, 0.125, 0.125, 0.0, 0.0],
    "tpot": 0.05,
    "e2e": 1.0,
}


@case("the worked example: six tokens in three chunks between two empty chunks", public=True)
def worked_example(ctx):
    # Rejects start_after_open (ttft 0.5), first_chunk_ttft (0.5), one_time_per_chunk
    # (three tokens), and e2e_at_stream_end (1.125).
    check_probe(ctx, EXAMPLE, "probe on the example stream", start=2.0, want=EXAMPLE_RECORD)


@case("the same stream with a sink that takes 1/32 s per chunk: the record is unchanged",
      public=True)  # fmt: skip
def slow_sink(ctx):
    # Rejects stamp_after_sink: every chunk still arrives when the server sent it.
    check_probe(
        ctx, EXAMPLE, "probe with a slow sink", start=2.0, sink_s=0.03125, want=EXAMPLE_RECORD
    )


@case("streams that fail or carry nothing: refused, broken early, broken late, empty",
      public=True)  # fmt: skip
def failures(ctx):
    # Rejects error_is_success.
    refused = script(0.5, [], 0.5, fail="open")
    check_probe(ctx, refused, "a refused request", start=1.0, want=record("error", 1.0, []))
    early = script(0.25, [(0.5, 0)], 0.75, fail="stream")
    check_probe(ctx, early, "a stream that breaks before any token", want=record("error", 0.0, []))
    late = script(0.25, [(0.5, 0), (0.75, 2), (1.0, 1)], 1.5, fail="stream")
    check_probe(
        ctx, late, "a stream that breaks after three tokens", start=4.0,
        want=record("error", 4.0, [4.75, 4.75, 5.0]),
    )  # fmt: skip
    empty = script(0.25, [(0.5, 0), (0.625, 0)], 0.75)
    check_probe(ctx, empty, "a stream of empty chunks", start=1.0, want=record("empty", 1.0, []))
    nothing = script(0.25, [], 0.5)
    check_probe(ctx, nothing, "a stream with no chunks", want=record("empty", 0.0, []))


SUMMARY_EXAMPLE = [
    record("ok", 0.0, [0.25, 0.375, 0.5]),
    record("ok", 1.0, [1.5, 1.5, 1.75, 2.25]),
    record("error", 2.0, [2.25]),
    record("ok", 3.0, [3.125]),
    record("empty", 4.0, []),
    record("ok", 5.0, [6.0, 6.0625]),
]


@case("summarize: the worked example, four ok records out of six", public=True)
def summary_example(ctx):
    # Rejects nearest_rank, itl_per_request_mean, and summary_counts_failures.
    check_summary(
        ctx, SUMMARY_EXAMPLE, 50, "summarize(results, 50)",
        {"requests": 6, "ok": 4, "ttft": 0.375, "tpot": 0.125, "itl": 0.125, "e2e": 0.78125},
    )  # fmt: skip
    check_summary(
        ctx, SUMMARY_EXAMPLE, 90, "summarize(results, 90)",
        {"requests": 6, "ok": 4, "ttft": 0.85, "tpot": 0.225, "itl": 0.375, "e2e": 1.19375},
    )  # fmt: skip


def simple(duration, fail=None):
    """A request that lasts `duration`: one token at half time, two at three quarters."""
    return script(duration / 4, [(duration / 2, 1), (3 * duration / 4, 2)], duration, fail)


@case("closed loop: five requests, two at a time; a freed slot is refilled at once",
      public=True)  # fmt: skip
def closed_loop_example(ctx):
    # Rejects waves (request 2 would wait for request 0), unbounded_concurrency, and
    # completion_order.
    scripts = [simple(d) for d in (1.0, 0.5, 0.25, 1.0, 0.5)]
    check_loop(
        ctx, scripts, 2, "run_closed_loop on the example", want_starts=[0.0, 0.0, 0.5, 0.75, 1.0]
    )


@case("one chunk holding every token, a single token, and tokens only in the last chunk")
def chunk_shapes(ctx):
    # Rejects one_time_per_chunk and first_chunk_ttft.
    check_probe(ctx, script(0.125, [(0.5, 7)], 0.5), "one chunk of seven tokens")
    check_probe(ctx, script(0.125, [(0.25, 0), (0.5, 1)], 0.75), "a single token", start=3.0)
    check_probe(
        ctx, script(0.0, [(0.125, 0), (0.25, 0), (0.375, 0), (1.0, 4)], 1.0),
        "three empty chunks, then four tokens",
    )  # fmt: skip
    check_probe(
        ctx, script(0.0, [(0.0, 1), (0.0, 2), (0.5, 1)], 0.5), "tokens the instant it is sent"
    )
    check_probe(
        ctx, script(0.5, [(0.5, 1), (0.75, 1), (2.0, 0), (3.0, 0)], 4.0),
        "two tokens, then a long tail of empty chunks",
    )  # fmt: skip


@case("a sink slower than the stream: chunks wait for the client and are stamped when read")
def sink_slower_than_stream(ctx):
    s = script(0.125, [(0.25, 1), (0.3125, 2), (0.375, 1), (0.4375, 0), (0.5, 3)], 0.5625)
    for sink_s in (0.0625, 0.125, 0.25):
        check_probe(
            ctx, s, f"probe with a sink of {sink_s:g} s per chunk", start=1.0, sink_s=sink_s
        )
    check_probe(
        ctx, script(0.125, [(0.25, 2), (0.3125, 2)], 0.375, fail="stream"),
        "a broken stream read through a slow sink", sink_s=0.25,
    )  # fmt: skip


def random_script(rng, fail_rate=0.0):
    tick = 1 / 1024
    connect = int(rng.integers(0, 200)) * tick
    offset = connect
    chunks = []
    for _ in range(int(rng.integers(0, 12))):
        offset += int(rng.integers(0, 120)) * tick
        chunks.append((offset, int(rng.choice([0, 0, 1, 1, 1, 2, 3, 5]))))
    close = offset + int(rng.integers(0, 60)) * tick
    fail = None
    if rng.random() < fail_rate:
        fail = "open" if rng.random() < 0.4 else "stream"
    return script(connect, chunks, close, fail)


@case("random streams: empty chunks, bursts, failures, with and without a slow sink")
def random_streams(ctx):
    rng = ctx.rng
    for i in range(40):
        s = random_script(rng, fail_rate=0.25)
        start = int(rng.integers(0, 4096)) / 1024
        sink_s = float(rng.choice([0.0, 0.0, 1 / 1024, 16 / 1024, 100 / 1024]))
        check_probe(ctx, s, f"probe on random stream {i}", start=start, sink_s=sink_s)


@case("closed loop: concurrency 1, concurrency above n, ties, and requests that fail")
def closed_loop_edges(ctx):
    # A failed request frees its slot like any other; rejects waves and
    # unbounded_concurrency again, and completion_order.
    durations = (0.5, 0.25, 0.25, 1.0, 0.125, 0.5, 0.25)
    plain = [simple(d) for d in durations]
    for concurrency in (1, 2, 3, 7, 50):
        check_loop(ctx, plain, concurrency, f"run_closed_loop, concurrency {concurrency}")
    mixed = [simple(d, fail) for d, fail in zip(
        durations, (None, "stream", None, "open", None, None, "open"), strict=True
    )]  # fmt: skip
    check_loop(ctx, mixed, 2, "run_closed_loop with failing requests")
    check_loop(ctx, mixed, 3, "run_closed_loop with failing requests and a slow sink",
               sink_s=0.03125)  # fmt: skip
    check_loop(ctx, [simple(0.5)], 4, "run_closed_loop with a single request")
    check_loop(ctx, [], 3, "run_closed_loop with no requests")


@case("closed loop on random streams, several concurrencies")
def closed_loop_random(ctx):
    rng = ctx.rng
    for i in range(6):
        scripts = [random_script(rng, fail_rate=0.15) for _ in range(int(rng.integers(5, 30)))]
        concurrency = int(rng.integers(1, 9))
        check_loop(ctx, scripts, concurrency, f"run_closed_loop, random workload {i}")


def random_records(rng, n):
    records = []
    for _ in range(n):
        t_start = int(rng.integers(0, 10_000)) / 64
        kind = rng.random()
        count = int(rng.choice([1, 1, 2, 5, 20, 60]))
        gaps = rng.integers(0, 40, size=count) / 256
        times = (t_start + int(rng.integers(1, 300)) / 256 + np.cumsum(gaps)).tolist()
        if kind < 0.1:
            records.append(record("error", t_start, times[: count // 2]))
        elif kind < 0.15:
            records.append(record("empty", t_start, []))
        else:
            records.append(record("ok", t_start, times))
    return records


@case("summarize: against numpy.percentile, pooled gaps, and records without latencies")
def summary_sweep(ctx):
    rng = ctx.rng
    for i in range(8):
        records = random_records(rng, int(rng.integers(3, 120)))
        ok = [r for r in records if r["status"] == "ok"]
        for q in (0, 50, 90, 95, 99, 99.9, 100):
            check_summary(ctx, records, q, f"summarize(random records {i}, {q})")
        if ok:
            got = ctx.call("summarize", [dict(r) for r in records], 95)
            gaps = [gap for r in ok for gap in r["itl"]]
            if gaps:
                check.allclose(
                    got["itl"], float(np.percentile(gaps, 95)), rtol=1e-9, atol=1e-9,
                    label="the p95 of all inter-token gaps",
                )  # fmt: skip
            check.allclose(
                got["ttft"], float(np.percentile([r["ttft"] for r in ok], 95)),
                rtol=1e-9, atol=1e-9, label="the p95 of ttft",
            )  # fmt: skip
    # Nothing to summarize, and nothing but one-token responses.
    check_summary(ctx, [], 50, "summarize of no records")
    failures_only = [record("error", 0.0, [0.5]), record("empty", 1.0, [])]
    check_summary(ctx, failures_only, 99, "summarize of failed and empty records only")
    one_token = [record("ok", float(i), [i + 0.25 * (i + 1)]) for i in range(5)]
    check_summary(ctx, one_token, 75, "summarize of one-token responses")
    kept = [dict(r) for r in SUMMARY_EXAMPLE]
    ctx.solution.summarize(kept, 50)
    check.that(kept == SUMMARY_EXAMPLE, "summarize must not modify the records")
