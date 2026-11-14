"""Tests for admission-control.

Two kinds of cases. `play` makes a hand-written sequence of admit / start / finish
calls and compares every decision and the three counters with literal values.
`simulate` puts the Admission object in front of a one-slot server on a simulated
clock, drives it with a seeded workload, and checks what admission control is for:
the queue stays within its bounds, accepted requests finish within the latency limit
that the bounds imply, and every decision equals the oracle's.

The server drains 1024 prompt tokens per second and arrivals fall on a 1/1024 s grid,
so every time in the simulation is exact.
"""

from whetstone.judge import case, check
from whetstone.kit import SimRequest, workloads

DRAIN = 1024.0  # prompt tokens per second, a power of two


def req(ident, prompt_tokens):
    return SimRequest(ident, 0.0, prompt_tokens, 1)


def decision_of(ctx, result, label, detail=""):
    """"admitted", or the retry_after_s of a rejection; checks the result's type."""
    solution = ctx.solution
    check.that(
        isinstance(result, (solution.Admitted, solution.Rejected)),
        f"{label}: admit must return an Admitted or a Rejected object",
        detail + f"it returned {result!r}\n",
    )
    if isinstance(result, solution.Admitted):
        return "admitted"
    retry = result.retry_after_s
    check.that(
        retry is None or (isinstance(retry, (int, float)) and not isinstance(retry, bool)),
        f"{label}: Rejected.retry_after_s must be a number or None",
        detail + f"retry_after_s = {retry!r}\n",
    )
    return retry


def same_decision(got, want):
    if isinstance(got, str) or isinstance(want, str) or got is None or want is None:
        return got == want
    return abs(got - want) <= 1e-9


def show(decision):
    if decision == "admitted":
        return "Admitted()"
    return f"Rejected(retry_after_s={decision!r})"


def play(ctx, limits, steps, label):
    """Make the calls in `steps` and compare each outcome with the expected one.

    ("admit", id, prompt_tokens, expected)   expected: "admitted", a float, or None
    ("start", id) / ("finish", id)
    ("state", queued_requests, queued_tokens, running_requests)
    """
    max_requests, max_tokens, drain = limits
    adm = ctx.solution.Admission(max_requests, max_tokens, drain)
    requests, log = {}, [f"Admission({max_requests}, {max_tokens}, {drain})"]
    for step in steps:
        kind = step[0]
        if kind == "admit":
            _, ident, tokens, want = step
            requests[ident] = req(ident, tokens)
            call = f"admit({ident}: {tokens} tokens)"
            got = decision_of(ctx, adm.admit(requests[ident]), f"{label}: {call}", "\n".join(log))
            log.append(f"  {call} -> {show(got)}")
            check.that(
                same_decision(got, want),
                f"{label}: admit returned the wrong decision",
                "\n".join(log) + f"\nexpected: {show(want)}\n",
            )
        elif kind in ("start", "finish"):
            getattr(adm, kind)(requests[step[1]])
            log.append(f"  {kind}({step[1]})")
        else:
            got = (adm.queued_requests, adm.queued_tokens, adm.running_requests)
            log.append(f"  (queued_requests, queued_tokens, running_requests) = {got}")
            check.that(
                got == tuple(step[1:]),
                f"{label}: queued_requests, queued_tokens, or running_requests is wrong",
                "\n".join(log) + f"\nexpected: {tuple(step[1:])}\n",
            )


class Sim:
    """The outcome of one simulation of a one-slot server behind an Admission object."""

    def __init__(self):
        self.decisions = []  # (request id, "admitted" or retry_after_s), in arrival order
        self.latencies = {}  # id -> finish time - arrival time, accepted requests only
        self.max_queued_requests = 0
        self.max_queued_tokens = 0
        self.counter_errors = []


def simulate(ctx, admission_class, requests, limits, *, pick="fifo", solution=True):
    """Run `requests` through admission control and a server with one slot.

    The server runs one request at a time for prompt_tokens / DRAIN seconds. When it
    becomes free it starts the oldest queued request (pick="fifo") or the one with the
    shortest prompt (pick="shortest"). At one instant, a completion and the start that
    follows it happen before the arrivals of that instant.
    """
    sim = ctx.simclock()
    out = Sim()
    adm = admission_class(*limits)
    queue, running = [], []

    def observe(where):
        tokens = sum(r.prompt_tokens for r in queue)
        out.max_queued_requests = max(out.max_queued_requests, len(queue))
        out.max_queued_tokens = max(out.max_queued_tokens, tokens)
        got = (adm.queued_requests, adm.queued_tokens, adm.running_requests)
        want = (len(queue), tokens, len(running))
        if got != want and len(out.counter_errors) < 3:
            out.counter_errors.append(f"t={sim.now():g} after {where}: got {got}, expected {want}")

    def start_next():
        if running or not queue:
            return
        request = min(queue, key=lambda r: r.prompt_tokens) if pick == "shortest" else queue[0]
        queue.remove(request)
        running.append(request)
        adm.start(request)
        observe(f"start({request.id})")
        sim.call_later(request.prompt_tokens / DRAIN, done, request, priority=-1)

    def done(request):
        running.remove(request)
        adm.finish(request)
        out.latencies[request.id] = sim.now() - request.arrival_s
        observe(f"finish({request.id})")
        start_next()

    def arrive(request):
        result = adm.admit(request)
        label = f"admit(request {request.id})"
        decision = decision_of(ctx, result, label) if solution else (
            "admitted" if not hasattr(result, "retry_after_s") else result.retry_after_s
        )
        out.decisions.append((request.id, decision))
        if decision == "admitted":
            queue.append(request)
        observe(label)
        start_next()

    workloads.drive(sim, requests, arrive)
    sim.run()
    return out


def check_simulation(ctx, requests, limits, label, *, pick="fifo"):
    max_requests, max_tokens, drain = limits
    out = simulate(ctx, ctx.solution.Admission, requests, limits, pick=pick)
    want = simulate(ctx, ctx.oracle.Admission, requests, limits, pick=pick, solution=False)
    admitted = sum(d == "admitted" for _, d in out.decisions)
    summary = (
        f"Admission({max_requests}, {max_tokens}, {drain}); {len(requests)} requests, "
        f"{admitted} admitted; the server drains {DRAIN:g} prompt tokens per second\n"
    )
    check.at_most("max_queued_requests", out.max_queued_requests, max_requests, unit="requests")
    check.at_most("max_queued_tokens", out.max_queued_tokens, max_tokens, unit="tokens")
    if pick == "fifo":
        # A request waits for the running request (at most the longest prompt) and
        # for what is queued up to and including itself (at most max_queued_tokens).
        limit_s = (max_tokens + max(r.prompt_tokens for r in requests)) / DRAIN
        check.at_most("max_accepted_latency", max(out.latencies.values()), limit_s, unit="s")
    check.that(
        not out.counter_errors,
        f"{label}: queued_requests, queued_tokens, or running_requests does not match the "
        "requests that are really queued and running",
        summary + "(queued_requests, queued_tokens, running_requests):\n"
        + "\n".join(out.counter_errors) + "\n",
    )
    for (ident, got), (_, expected) in zip(out.decisions, want.decisions, strict=True):
        check.that(
            same_decision(got, expected),
            f"{label}: a request got the wrong admission decision",
            summary + f"first difference: request {ident} "
            f"(prompt_tokens={requests[ident].prompt_tokens}, arrival_s={requests[ident].arrival_s})\n"
            f"expected: {show(expected)}\n     got: {show(got)}\n",
        )
    return out


EXAMPLE = [
    ("admit", "a", 400, "admitted"),
    ("admit", "b", 300, "admitted"),
    ("admit", "c", 400, 0.8),  # tokens: 700 + 400 > 1000; fits once a (400) has left
    ("admit", "d", 200, "admitted"),
    ("state", 3, 900, 0),
    ("admit", "e", 50, 0.8),  # requests: 3 + 1 > 3; fits once a has left
    ("admit", "f", 700, 1.4),  # fits only once a and b (700 tokens) have left
    ("admit", "g", 1200, None),  # can never fit
    ("state", 3, 900, 0),
    ("start", "a"),
    ("state", 2, 500, 1),
    ("admit", "c", 400, "admitted"),
    ("state", 3, 900, 1),
    ("finish", "a"),
    ("state", 3, 900, 0),
]


@case("the worked example: at most 3 requests and 1000 tokens queued", public=True)
def worked_example(ctx):
    play(ctx, (3, 1000, 500.0), EXAMPLE, "the worked example")


@case("a request that fills the queue exactly is admitted; one more is not", public=True)
def exact_fit(ctx):
    # Rejects rejects_exact_fit and check_after_enqueue (the rejected request must
    # leave no trace, so the queue still has room once one request has started).
    steps = [
        ("admit", 1, 60, "admitted"),
        ("admit", 2, 40, "admitted"),  # exactly 2 requests and exactly 100 tokens
        ("state", 2, 100, 0),
        ("admit", 3, 1, 0.9375),  # both bounds are full; request 1 (60 tokens) must leave
        ("admit", 4, 1, 0.9375),
        ("state", 2, 100, 0),
        ("start", 1),
        ("state", 1, 40, 1),
        ("admit", 5, 60, "admitted"),  # exactly full again
        ("admit", 6, 60, 1.5625),  # 2 and 5 must both leave: 60 + 60 > 100 with 5 queued
        ("state", 2, 100, 1),
    ]
    play(ctx, (2, 100, 64.0), steps, "at most 2 requests and 100 tokens")


@case("running requests do not count against the queue bounds", public=True)
def running_is_not_queued(ctx):
    # Rejects running_counted_as_queued.
    steps = [
        ("admit", "p", 500, "admitted"),
        ("admit", "q", 500, "admitted"),
        ("start", "p"),
        ("start", "q"),
        ("state", 0, 0, 2),
        ("admit", "r", 1000, "admitted"),  # the whole token budget is free again
        ("state", 1, 1000, 2),
        ("finish", "p"),
        ("state", 1, 1000, 1),
        ("admit", "s", 1, 1.0),  # r must leave: 1000 tokens at 1000 tokens/s
        ("start", "r"),
        ("finish", "q"),
        ("finish", "r"),
        ("state", 0, 0, 0),
    ]
    play(ctx, (4, 1000, 1000.0), steps, "at most 4 requests and 1000 tokens")


@case("3x overload: the queue stays bounded and accepted requests finish in time", public=True)
def overload(ctx):
    # Rejects unbounded_queue (latency grows to minutes) and ignores_token_bound.
    # The server finishes about 1.9 requests per second; 6 arrive.
    requests = workloads.poisson(
        ctx.rng, rate=6.0, n=300, prompt_tokens=(64, 1024), quantum_s=2**-10
    )
    out = check_simulation(ctx, requests, (8, 4096, DRAIN), "3x overload")
    admitted = sum(d == "admitted" for _, d in out.decisions)
    check.that(
        admitted >= 80,
        "under overload the server should stay busy: too few requests were admitted",
        f"{admitted} of {len(requests)} requests were admitted\n",
    )


@case("retry_after_s: how many departures a request needs, for each bound")
def retry_estimates(ctx):
    # Rejects retry_whole_queue, retry_oldest_only, and oversized_gets_retry_time.
    steps = [
        ("admit", 1, 100, "admitted"),
        ("admit", 2, 200, "admitted"),
        ("admit", 3, 300, "admitted"),
        ("admit", 4, 150, "admitted"),
        ("state", 4, 750, 0),
        ("admit", 5, 50, 1.5625),  # request bound only: request 1 must leave (100 / 64)
        ("admit", 6, 450, 4.6875),  # token bound: 1 and 2 must leave (300 / 64)
        ("admit", 7, 800, 9.375),  # 1, 2, and 3 must leave (600 / 64)
        ("admit", 8, 1000, 11.71875),  # the whole queue must leave (750 / 64)
        ("admit", 9, 1001, None),  # more than max_queued_tokens: never
        ("state", 4, 750, 0),
    ]
    play(ctx, (4, 1000, 64.0), steps, "at most 4 requests and 1000 tokens")
    # A prompt of exactly max_queued_tokens is admitted into an empty queue.
    steps = [
        ("admit", 1, 1000, "admitted"),
        ("admit", 2, 1001, None),
        ("admit", 3, 1000, 15.625),
        ("start", 1),
        ("admit", 2, 1001, None),  # still never, also with an empty queue
        ("admit", 3, 1000, "admitted"),
    ]
    play(ctx, (4, 1000, 64.0), steps, "a prompt as large as the token bound")


@case("the server starts requests out of arrival order")
def out_of_order_starts(ctx):
    # Rejects start_pops_oldest.
    steps = [
        ("admit", "long", 700, "admitted"),
        ("admit", "short", 100, "admitted"),
        ("admit", "mid", 200, "admitted"),
        ("start", "short"),
        ("state", 2, 900, 1),
        ("admit", "x", 150, 7.0),  # long (700 tokens) is still the oldest in the queue
        ("start", "mid"),
        ("state", 1, 700, 2),
        ("admit", "y", 300, "admitted"),
        ("finish", "mid"),
        ("finish", "short"),
        ("start", "long"),
        ("state", 1, 300, 1),
    ]
    play(ctx, (3, 1000, 100.0), steps, "at most 3 requests and 1000 tokens")


@case("light load: nothing is rejected and nothing is delayed beyond its service time")
def light_load(ctx):
    requests = workloads.poisson(
        ctx.rng, rate=0.25, n=60, prompt_tokens=(64, 512), quantum_s=2**-10
    )
    out = check_simulation(ctx, requests, (4, 2048, DRAIN), "light load")
    rejected = [ident for ident, d in out.decisions if d != "admitted"]
    check.that(not rejected, "a request was rejected although the queue had room")


@case("bursts of simultaneous arrivals, with completions at the same instants")
def bursts(ctx):
    # Every burst arrives at a whole second, and with prompts that are multiples of
    # 128 tokens the completions fall on the same grid: ties between a completion,
    # the next start, and arrivals occur throughout.
    rng = ctx.rng
    requests = workloads.bursty(rng, bursts=30, burst_size=6, period_s=1.0, prompt_tokens=128)
    sizes = rng.integers(1, 9, size=len(requests)) * 128
    requests = workloads.replay(
        [(r.arrival_s, int(n), 1) for r, n in zip(requests, sizes, strict=True)]
    )
    check_simulation(ctx, requests, (5, 2048, DRAIN), "bursts of 6 every second")


@case("a shortest-prompt-first server under overload, tight token bound")
def shortest_first_overload(ctx):
    requests = workloads.poisson(
        ctx.rng, rate=8.0, n=250, prompt_tokens=(32, 1024), quantum_s=2**-10
    )
    check_simulation(ctx, requests, (16, 1536, DRAIN), "shortest first", pick="shortest")
    requests = workloads.poisson(
        ctx.rng, rate=8.0, n=250, prompt_tokens=(32, 1024), quantum_s=2**-10
    )
    check_simulation(ctx, requests, (2, 100_000, DRAIN), "request bound only")
