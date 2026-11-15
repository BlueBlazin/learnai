"""Tests for retry-backoff-client.

A scripted server answers each attempt after a fixed duration with a status and
headers, or breaks the connection. The test records the virtual time of every send
and how the call ended. Hand-checked scripts use binary fractions throughout, so
their send times are exact and are compared with literals; everything else is
compared with the oracle run through the same simulation, to 1e-9 s.
"""

from whetstone.judge import CheckFailure, case, check

TOL = 1e-9


class Response:
    def __init__(self, status, headers=None):
        self.status = status
        self.headers = dict(headers or {})

    def __repr__(self):
        return f"Response({self.status}, {self.headers!r})"


class Budget:
    """The retry budget of the statement (gRPC's retry throttling): failures drain a
    bucket, successes refill it by a fraction, retries need it more than half full."""

    def __init__(self, max_tokens=10, token_ratio=0.5):
        self.max_tokens, self.token_ratio = max_tokens, token_ratio
        self.tokens = float(max_tokens)
        self.calls = []

    def failure(self):
        self.calls.append("failure")
        self.tokens = max(0.0, self.tokens - 1.0)

    def success(self):
        self.calls.append("success")
        self.tokens = min(float(self.max_tokens), self.tokens + self.token_ratio)

    def allows_retry(self):
        return self.tokens > self.max_tokens / 2


def ok(duration=0.0):
    return (duration, Response(200))


def status(code, duration=0.0, **headers):
    return (duration, Response(code, {name.replace("_", "-"): v for name, v in headers.items()}))


def broken(duration=0.0):
    return (duration, None)


class Run:
    def __init__(self):
        self.sends = []  # virtual time of each send
        self.outcome = None  # ("returned", status) or ("raised", "ConnectionError")
        self.ended = None

    def text(self):
        return f"sends at {self.sends!r}; {self.outcome}; ended at {self.ended!r}"


def scripted_send(sim, script, run):
    """An async `send` that plays `script` attempt by attempt, then answers 200."""

    async def send():
        attempt = len(run.sends)
        run.sends.append(sim.now())
        if attempt > 200:
            raise CheckFailure("the client sent the request more than 200 times")
        duration, response = script[attempt] if attempt < len(script) else ok()
        if duration:
            await sim.sleep(duration)
        if response is None:
            raise ConnectionError("connection reset by peer")
        return response

    return send


async def one_call(sim, module, send, policy_args, uniforms, budget, run):
    policy = module.RetryPolicy(**policy_args)
    try:
        result = await module.call_with_retry(send, policy, list(uniforms), budget)
    except ConnectionError:
        run.outcome = ("raised", "ConnectionError")
    else:
        if not isinstance(result, Response):
            raise CheckFailure(
                "call_with_retry must return the response object that send returned",
                f"it returned {result!r}\n",
            )
        run.outcome = ("returned", result.status)
    run.ended = sim.now()


def play(ctx, module, script, policy_args, uniforms, *, start=0.0, budget=None):
    sim = ctx.simclock()
    run = Run()
    send = scripted_send(sim, script, run)
    sim.spawn_at(start, lambda: one_call(sim, module, send, policy_args, uniforms, budget, run))
    sim.run()
    if run.outcome is None:
        raise CheckFailure("call_with_retry never returned", f"sends at {run.sends!r}\n")
    return run


def describe(script, policy_args, uniforms):
    lines = [f"RetryPolicy({', '.join(f'{k}={v!r}' for k, v in policy_args.items())})"]
    lines.append(f"uniforms = {list(uniforms)[:8]!r}")
    for i, (duration, response) in enumerate(script):
        what = "the connection breaks" if response is None else repr(response)
        lines.append(f"attempt {i + 1}: {what} after {duration:g} s")
    lines.append("later attempts: Response(200, {})")
    return "\n".join(lines) + "\n"


def same_times(got, want):
    return len(got) == len(want) and all(abs(g - w) <= TOL for g, w in zip(got, want, strict=True))


def compare(got, want_sends, want_outcome, label, context):
    detail = (
        context
        + f"expected: sends at {want_sends!r}; {want_outcome}\n"
        + f"     got: sends at {got.sends!r}; {got.outcome}\n"
    )
    check.that(
        len(got.sends) == len(want_sends),
        f"{label}: the request was sent {len(got.sends)} times, which is not the specified number",
        detail,
    )
    check.that(
        same_times(got.sends, want_sends), f"{label}: an attempt was sent at the wrong time", detail
    )
    check.that(got.outcome == want_outcome, f"{label}: the call ended the wrong way", detail)


def check_call(ctx, script, policy_args, uniforms, label, *, start=0.0, sends=None, outcome=None):
    got = play(ctx, ctx.solution, script, policy_args, uniforms, start=start)
    if sends is None:
        want = play(ctx, ctx.oracle, script, policy_args, uniforms, start=start)
        sends, outcome = want.sends, want.outcome
    compare(got, sends, outcome, label, describe(script, policy_args, uniforms))


POLICY = {"max_attempts": 4, "base_s": 0.5, "cap_s": 8.0, "deadline_s": 30.0}


@case("the worked example: two 503s, then success; delays 0.25 s and 0.75 s", public=True)
def worked_example(ctx):
    # Rejects no_jitter (sends at 1.5, 2.5), equal_jitter, exponent_from_one.
    check_call(
        ctx, [status(503), status(503), ok()], POLICY, [0.5, 0.75, 0.5],
        "two 503 responses, then 200", start=1.0, sends=[1.0, 1.25, 2.0], outcome=("returned", 200),
    )  # fmt: skip


@case("which failures are retried: 429 and 503 yes; 400, 404, 500 no", public=True)
def retryable_statuses(ctx):
    # Rejects retries_client_errors.
    for code in (400, 401, 404, 422, 500, 502):
        check_call(
            ctx, [status(code), ok()], POLICY, [0.5, 0.5], f"a {code} response",
            sends=[0.0], outcome=("returned", code),
        )  # fmt: skip
    check_call(
        ctx, [status(429), ok()], POLICY, [0.5, 0.5], "a 429 response",
        sends=[0.0, 0.25], outcome=("returned", 200),
    )  # fmt: skip
    check_call(
        ctx, [ok()], POLICY, [0.5], "an immediate 200", sends=[0.0], outcome=("returned", 200)
    )
    check_call(
        ctx, [status(503), status(404), ok()], POLICY, [0.5, 0.5], "a 503, then a 404",
        sends=[0.0, 0.25], outcome=("returned", 404),
    )  # fmt: skip


@case("Retry-After is honored exactly, whatever its case, and uses no uniform", public=True)
def retry_after(ctx):
    # Rejects ignores_retry_after, retry_after_case_sensitive, and
    # uniform_spent_on_retry_after (the 503 after the 429 must use uniforms[0]).
    check_call(
        ctx, [status(429, Retry_After="2"), ok()], POLICY, [0.5], "429 with Retry-After: 2",
        sends=[0.0, 2.0], outcome=("returned", 200),
    )  # fmt: skip
    check_call(
        ctx, [status(503, retry_after="3"), ok()], POLICY, [0.5], "503 with retry-after: 3",
        sends=[0.0, 3.0], outcome=("returned", 200),
    )  # fmt: skip
    check_call(
        ctx, [status(429, Retry_After="2"), status(503), ok()], POLICY, [0.25, 0.75],
        "Retry-After, then a plain 503: the backoff is the second retry and uses uniforms[0]",
        sends=[0.0, 2.0, 2.25], outcome=("returned", 200),
    )  # fmt: skip
    check_call(
        ctx, [status(429, Retry_After="20"), ok()], POLICY, [0.5],
        "Retry-After longer than cap_s is still honored",
        sends=[0.0, 20.0], outcome=("returned", 200),
    )  # fmt: skip


@case("the limits: max_attempts sends, the capped ceiling, and the total deadline", public=True)
def limits(ctx):
    # Rejects one_attempt_too_many, cap_after_jitter, deadline_ignored, and
    # deadline_before_sleep.
    always_503 = [status(503)] * 12
    check_call(
        ctx, always_503, POLICY, [0.5] * 8, "a server that always answers 503",
        sends=[0.0, 0.25, 0.75, 1.75], outcome=("returned", 503),
    )  # fmt: skip
    capped = {"max_attempts": 7, "base_s": 1.0, "cap_s": 4.0, "deadline_s": 100.0}
    check_call(
        ctx, always_503, capped, [0.5] * 8, "ceilings 1, 2, 4, 4, 4, 4 with a cap of 4 s",
        sends=[0.0, 0.5, 1.5, 3.5, 5.5, 7.5, 9.5], outcome=("returned", 503),
    )  # fmt: skip
    tight = {"max_attempts": 10, "base_s": 1.0, "cap_s": 8.0, "deadline_s": 4.0}
    check_call(
        ctx, always_503, tight, [0.5] * 12, "a 4 s deadline: the retry due at 7.5 s is not made",
        sends=[0.0, 0.5, 1.5, 3.5], outcome=("returned", 503),
    )  # fmt: skip
    exact = {"max_attempts": 10, "base_s": 1.0, "cap_s": 8.0, "deadline_s": 3.5}
    check_call(
        ctx, always_503, exact, [0.5] * 12, "a retry due exactly at the deadline is not made",
        sends=[0.0, 0.5, 1.5], outcome=("returned", 503),
    )  # fmt: skip
    check_call(
        ctx, [status(429, Retry_After="5"), ok()], tight, [0.5],
        "Retry-After beyond the deadline: give up at once",
        sends=[0.0], outcome=("returned", 429),
    )  # fmt: skip


@case("connection errors: retried for an idempotent request, raised for any other",
      public=True)  # fmt: skip
def connection_errors(ctx):
    # Rejects retries_non_idempotent.
    check_call(
        ctx, [broken(0.5), ok()], POLICY, [0.5], "an idempotent request whose connection breaks",
        sends=[0.0, 0.75], outcome=("returned", 200),
    )  # fmt: skip
    once = {**POLICY, "idempotent": False}
    check_call(
        ctx, [broken(0.5), ok()], once, [0.5], "a non-idempotent request whose connection breaks",
        sends=[0.0], outcome=("raised", "ConnectionError"),
    )  # fmt: skip
    check_call(
        ctx, [status(503), ok()], once, [0.5], "a non-idempotent request answered 503",
        sends=[0.0, 0.25], outcome=("returned", 200),
    )  # fmt: skip
    check_call(
        ctx, [broken()] * 6, POLICY, [0.5] * 6, "a connection that always breaks",
        sends=[0.0, 0.25, 0.75, 1.75], outcome=("raised", "ConnectionError"),
    )  # fmt: skip


def herd(ctx, module, clients, uniforms, policy_args, recover_s):
    """`clients` calls start at t = 0 against a server that answers 503 until
    `recover_s`; an attempt takes 1/64 s. Returns every send time."""
    sim = ctx.simclock()
    sends = []

    def make_send():
        async def send():
            sends.append(sim.now())
            if len(sends) > 5000:
                raise CheckFailure("the clients sent more than 5000 requests")
            overloaded = sim.now() < recover_s
            await sim.sleep(2**-6)
            return Response(503 if overloaded else 200)

        return send

    async def client(i):
        policy = module.RetryPolicy(**policy_args)
        await module.call_with_retry(make_send(), policy, list(uniforms[i]))

    for i in range(clients):
        sim.spawn(client(i))
    sim.run()
    return sorted(sends)


def busiest_window(times, width):
    """The largest number of sends in any half-open window of `width` seconds."""
    best = lo = 0
    for hi, t in enumerate(times):
        while times[lo] <= t - width:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


@case("thundering herd: 64 clients fail together; at most 24 retries in any 1/8 s",
      public=True)  # fmt: skip
def thundering_herd(ctx):
    # The bound: 24. Measured: the reference has at most 12 retries in any 1/8 s;
    # no_jitter has all 64 in one instant.
    policy_args = {"max_attempts": 6, "base_s": 2.0, "cap_s": 16.0, "deadline_s": 120.0}
    uniforms = ctx.rng.random((64, 8))
    sends = herd(ctx, ctx.solution, 64, uniforms, policy_args, recover_s=3.0)
    want = herd(ctx, ctx.oracle, 64, uniforms, policy_args, recover_s=3.0)
    retries = [t for t in sends if t > 0.0]
    check.that(len(retries) > 0, "no client retried after the outage began")
    check.at_most("retries_in_busiest_125ms", busiest_window(retries, 0.125), 24, unit="requests")
    check.that(
        same_times(sends, want),
        "thundering herd: the clients' send times are not the specified ones",
        f"expected (first 12 after t = 0): {[t for t in want if t > 0][:12]!r}\n"
        f"     got (first 12 after t = 0): {retries[:12]!r}\n",
    )


def budget_run(ctx, module, calls, script_for, policy_args, uniforms, gap_s):
    """Sequential calls, one every `gap_s`, that share one budget."""
    sim = ctx.simclock()
    budget = Budget()
    runs = [Run() for _ in range(calls)]
    for i in range(calls):
        send = scripted_send(sim, script_for(i), runs[i])
        sim.spawn_at(
            i * gap_s,
            lambda i=i, send=send: one_call(
                sim, module, send, policy_args, uniforms[i], budget, runs[i]
            ),
        )
    sim.run()
    return runs, budget


@case("a shared retry budget: during an outage the retries stop, afterwards they resume",
      public=True)  # fmt: skip
def retry_budget(ctx):
    # Rejects budget_ignored (4 sends per call throughout the outage) and
    # budget_never_refilled (no retry for the last call).
    policy_args = dict(POLICY)
    uniforms = ctx.rng.random((40, 4))

    def script_for(i):
        if i < 20:
            return [status(503)] * 4  # the outage: every attempt fails
        if i == 39:
            return [status(503), ok()]  # one transient failure, long after
        return [ok()]

    got, budget = budget_run(ctx, ctx.solution, 40, script_for, policy_args, uniforms, 16.0)
    want, want_budget = budget_run(ctx, ctx.oracle, 40, script_for, policy_args, uniforms, 16.0)
    total = sum(len(run.sends) for run in got)
    check.at_most("sends_for_40_calls", total, 50, unit="requests")
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        compare(g, w.sends, w.outcome, f"call {i} of 40 sharing a budget", "")
    check.that(
        budget.calls == want_budget.calls,
        "the budget was not told about the attempts as specified",
        f"expected (first 30): {want_budget.calls[:30]!r}\n"
        f"     got (first 30): {budget.calls[:30]!r}\n",
    )


@case("Retry-After values that are not delay-seconds fall back to backoff")
def retry_after_values(ctx):
    for value, label in (
        ("0", "zero seconds: resend at once"),
        (" 3 ", "surrounding spaces"),
        ("007", "leading zeros"),
        ("abc", "not a number"),
        ("-1", "negative"),
        ("1.5", "a decimal fraction"),
        ("", "empty"),
        ("Wed, 21 Oct 2026 07:28:00 GMT", "an HTTP-date"),
    ):
        for name in ("Retry-After", "retry-after", "RETRY-AFTER"):
            script = [
                (0.125, Response(429, {name: value, "Content-Type": "application/json"})),
                ok(),
            ]
            check_call(ctx, script, POLICY, [0.75, 0.5], f"{name}: {value!r} ({label})")
    script = [(0.0, Response(503, {"X-Retry-After": "9", "Retry": "4"})), ok()]
    check_call(ctx, script, POLICY, [0.75, 0.5], "headers that only resemble Retry-After")


@case("attempts that take time: the deadline counts from the call, the delay from the response")
def slow_attempts(ctx):
    # Rejects deadline_before_sleep and deadline_ignored.
    for deadline_s in (2.0, 2.75, 3.0, 4.0, 4.25, 6.0, 9.0, 50.0):
        policy_args = {"max_attempts": 8, "base_s": 1.0, "cap_s": 4.0, "deadline_s": deadline_s}
        script = [status(503, duration=0.5), broken(0.25), status(429, duration=1.0)] * 3
        check_call(
            ctx, script, policy_args, [0.5, 0.25, 0.75, 0.5, 0.5, 0.25, 0.5, 0.5],
            f"slow attempts under a deadline of {deadline_s} s", start=10.0,
        )  # fmt: skip
    check_call(ctx, [status(503)] * 3, {**POLICY, "max_attempts": 1}, [0.5], "max_attempts of 1")
    check_call(
        ctx, [status(503)] * 3, {**POLICY, "max_attempts": 2}, [0.0, 0.5],
        "a uniform of exactly 0: the retry is sent in the same instant",
    )  # fmt: skip


@case("random scripts: statuses, Retry-After, broken connections, policies, uniforms")
def random_scripts(ctx):
    rng = ctx.rng
    for i in range(120):
        script = []
        for _ in range(int(rng.integers(1, 9))):
            duration = int(rng.integers(0, 64)) / 64
            kind = rng.random()
            if kind < 0.35:
                script.append(status(503, duration))
            elif kind < 0.55:
                script.append(status(429, duration))
            elif kind < 0.7:
                name = str(rng.choice(["Retry-After", "retry-after"]))
                value = str(rng.choice(["0", "1", "2", "5", "x"]))
                script.append((duration, Response(int(rng.choice([429, 503])), {name: value})))
            elif kind < 0.85:
                script.append(broken(duration))
            else:
                script.append(status(int(rng.choice([200, 400, 404, 500])), duration))
        policy_args = {
            "max_attempts": int(rng.integers(1, 8)),
            "base_s": float(rng.choice([0.125, 0.5, 1.0, 3.0])),
            "cap_s": float(rng.choice([1.0, 4.0, 10.0])),
            "deadline_s": float(rng.choice([1.0, 5.0, 12.0, 100.0])),
            "idempotent": bool(rng.random() < 0.7),
        }
        uniforms = rng.random(10)
        check_call(
            ctx, script, policy_args, uniforms, f"random script {i}",
            start=float(int(rng.integers(0, 640)) / 64),
        )  # fmt: skip


@case("a budget shared by overlapping calls, with successes, client errors, and broken links")
def budget_accounting(ctx):
    # Rejects budget_ignored and budget_never_refilled; a 404 and a 500 touch the
    # budget in neither direction.
    rng = ctx.rng
    outcomes = [status(503), status(429), broken(), ok(), ok(), status(404), status(500)]
    for trial in range(6):
        scripts = [
            [outcomes[int(k)] for k in rng.integers(0, len(outcomes), size=5)] for _ in range(30)
        ]
        uniforms = rng.random((30, 6))
        policy_args = {**POLICY, "max_attempts": 5, "idempotent": bool(trial % 2)}
        args = (30, lambda i, scripts=scripts: scripts[i], policy_args, uniforms, 0.375)
        got, budget = budget_run(ctx, ctx.solution, *args)
        want, want_budget = budget_run(ctx, ctx.oracle, *args)
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            compare(g, w.sends, w.outcome, f"shared budget, trial {trial}, call {i}", "")
        check.that(
            sorted(budget.calls) == sorted(want_budget.calls),
            f"shared budget, trial {trial}: wrong number of success() or failure() reports",
            f"expected: {want_budget.calls.count('success')} successes and "
            f"{want_budget.calls.count('failure')} failures\n"
            f"     got: {budget.calls.count('success')} successes and "
            f"{budget.calls.count('failure')} failures\n",
        )
