"""Tests for token-bucket-limiter.

`simulate` plays a script of calls against a TokenBucket on a simulated clock and
records when each `acquire` returned, what each `try_acquire` answered, and which calls
raised. Hand-checked scripts are compared with literal times; every script is also
played against the oracle, and the two records must agree.

All rates are powers of two and all call times are multiples of 1/64 s, so every
balance and every grant time is exactly representable: a correct solution reproduces
the times to the last bit, and the 1e-9 s tolerance is only a courtesy.
"""

from learnai.judge import CheckFailure, case, check

HOPS = 4  # event-loop turns granted to an acquire() call before the next same-instant call
TOL_S = 1e-9


class Run:
    """What happened in one simulation."""

    def __init__(self, rate, burst, script):
        self.rate, self.burst, self.script = rate, burst, script
        self.log = []  # (time, text) for the failure detail
        self.grants = {}  # id -> time at which acquire() returned
        self.tries = {}  # id -> what try_acquire() returned
        self.errors = {}  # id -> time at which the call raised ValueError
        self.waiting = {}  # id -> n, acquire() calls that have not returned

    def timeline(self):
        lines = [f"TokenBucket(rate_per_s={self.rate}, burst={self.burst})", "timeline:"]
        lines += [f"  t={t:<9g} {text}" for t, text in self.log[:70]]
        if len(self.log) > 70:
            lines.append(f"  ... {len(self.log) - 70} more events")
        return "\n".join(lines) + "\n"


def simulate(ctx, bucket_class, rate, burst, script):
    """Play `script`, a list of calls sorted by time, against a new bucket.

    ("acquire", t, id, n)   start `await bucket.acquire(n)` at time t
    ("try", t, id, n)       call `bucket.try_acquire(n)` at time t
    ("refund", t, n)        call `bucket.refund(n)` at time t

    Calls at the same time are made in list order, and before any timer the bucket
    itself set for that instant (the statement's rule makes the outcome independent of
    that order; a solution that breaks the rule is caught because the test goes first).
    """
    sim = ctx.simclock()
    run = Run(rate, burst, script)
    holder = {}

    def note(text):
        run.log.append((sim.now(), text))

    async def client(ident, n):
        run.waiting[ident] = n
        note(f"acquire({n}) called            [{ident}]")
        try:
            result = await holder["bucket"].acquire(n)
        except ValueError:
            run.errors[ident] = sim.now()
            note(f"acquire({n}) raised ValueError  [{ident}]")
        else:
            if result is not None:
                raise CheckFailure("acquire must return None", f"it returned {result!r}\n")
            run.grants[ident] = sim.now()
            note(f"acquire({n}) returned           [{ident}]")
        del run.waiting[ident]

    def call(op):
        kind, _, *args = op
        bucket = holder["bucket"]
        if kind == "acquire":
            sim.spawn(client(*args))
        elif kind == "try":
            ident, n = args
            try:
                ok = bucket.try_acquire(n)
            except ValueError:
                run.errors[ident] = sim.now()
                note(f"try_acquire({n}) raised ValueError [{ident}]")
                return
            if not isinstance(ok, bool):
                raise CheckFailure("try_acquire must return a bool", f"it returned {ok!r}\n")
            run.tries[ident] = ok
            note(f"try_acquire({n}) -> {ok}        [{ident}]")
        else:
            (n,) = args
            note(f"refund({n})")
            bucket.refund(n)

    def run_group(group, i=0):
        call(group[i])
        if i + 1 < len(group):
            hop(HOPS, lambda: run_group(group, i + 1))

    def hop(turns, then):
        if turns == 0:
            then()
        else:
            sim.loop.call_soon(hop, turns - 1, then)

    def create():
        holder["bucket"] = bucket_class(rate, burst)

    sim.call_at(0.0, create, priority=-2)
    groups = {}
    for op in script:
        groups.setdefault(op[1], []).append(op)
    for when, group in groups.items():
        sim.call_at(when, run_group, group, priority=-1)

    # By this time every request of the script has been served by a correct bucket.
    asked = sum(op[3] for op in script if op[0] == "acquire" and op[3] <= burst)
    horizon = max(op[1] for op in script) + asked / rate + 8.0
    sim.run(until=horizon)
    return run


def check_same(run, grants, tries, errors, label):
    """Compare a run with expected grant times, try_acquire answers, and errors."""
    detail = run.timeline()
    for ident in errors:
        check.that(
            ident in run.errors,
            f"{label}: a call for more tokens than `burst` did not raise ValueError",
            detail + f"call [{ident}] should have raised ValueError\n",
        )
        check.that(
            abs(run.errors[ident] - errors[ident]) <= TOL_S,
            f"{label}: ValueError for an oversized request was not raised at the time of the call",
            detail + f"call [{ident}] was made at t={errors[ident]:g}\n",
        )
    check.that(
        set(run.errors) <= set(errors),
        f"{label}: a call raised ValueError although n <= burst",
        detail + f"calls that raised: {sorted(set(run.errors) - set(errors))}\n",
    )
    for ident, ok in tries.items():
        check.that(
            run.tries.get(ident) is ok,
            f"{label}: try_acquire returned the wrong answer",
            detail + f"try_acquire [{ident}] should return {ok}\n",
        )
    never = sorted(set(grants) - set(run.grants), key=str)
    check.that(
        not never,
        f"{label}: an acquire() call never returned",
        detail + f"still waiting when the simulation ended: {never}\n",
    )
    check.that(
        set(run.grants) == set(grants),
        f"{label}: an acquire() call returned that should not have",
        detail,
    )
    for ident, when in grants.items():
        got = run.grants[ident]
        check.that(
            abs(got - when) <= TOL_S,
            f"{label}: an acquire() call returned at the wrong time",
            detail + f"acquire [{ident}]: expected to return at t={when!r}, returned at t={got!r}\n",
        )


def check_against_oracle(ctx, rate, burst, script, label):
    run = simulate(ctx, ctx.solution.TokenBucket, rate, burst, script)
    want = simulate(ctx, ctx.oracle.TokenBucket, rate, burst, script)
    assert not want.waiting, (label, want.waiting)  # the script itself must be servable
    check_same(run, want.grants, want.tries, want.errors, label)
    return run


# The statement's worked example.
EXAMPLE = [
    ("acquire", 0.0, "A", 192),
    ("acquire", 0.0, "B", 128),
    ("acquire", 0.5, "C", 16),
    ("try", 0.5, "D", 8),
    ("refund", 1.125, 32),
    ("try", 1.5, "E", 40),
    ("try", 10.0, "F", 256),
    ("try", 10.0, "G", 1),
]


@case("the worked example: rate 64 tokens/s, burst 256", public=True)
def worked_example(ctx):
    # Rejects barging (C or D served at 0.5), refund_does_not_wake (C at 1.25),
    # tick_refill, starts_empty, and uncapped_refill (G succeeds at t=10).
    run = simulate(ctx, ctx.solution.TokenBucket, 64, 256, EXAMPLE)
    check_same(
        run,
        grants={"A": 0.0, "B": 1.0, "C": 1.125},
        tries={"D": False, "E": True, "F": True, "G": False},
        errors={},
        label="the worked example",
    )
    check_against_oracle(ctx, 64, 256, EXAMPLE, "the worked example")


@case("the bucket starts full, refills continuously, and never holds more than burst", public=True)
def refill_and_cap(ctx):
    # Rejects starts_empty, tick_refill (0.75 s is not a whole second), uncapped_refill.
    script = [
        ("try", 0.0, "full", 100),  # 100 of 100: the bucket starts full
        ("try", 0.0, "empty", 1),  # nothing left
        ("try", 0.25, "early", 5),  # 4 tokens accrued
        ("try", 0.75, "partial", 12),  # 12 accrued
        ("try", 0.75, "again", 1),
        ("try", 60.0, "capped", 100),  # a minute idle: still only 100
        ("try", 60.0, "over", 1),
        ("acquire", 60.5, "wait", 24),  # 8 there, 16 missing: one more second
    ]
    run = simulate(ctx, ctx.solution.TokenBucket, 16, 100, script)
    check_same(
        run,
        grants={"wait": 61.5},
        tries={
            "full": True,
            "empty": False,
            "early": False,
            "partial": True,
            "again": False,
            "capped": True,
            "over": False,
        },
        errors={},
        label="rate 16, burst 100",
    )


@case("first in, first out: a small request does not overtake a waiting large one", public=True)
def fifo_no_barging(ctx):
    # Rejects barging and serves_any_fitting_waiter.
    script = [
        ("try", 0.0, "drain", 64),
        ("acquire", 0.0, "big", 48),  # would be served at 6.0; with the refund, 5.0
        ("acquire", 1.0, "small", 2),  # 8 tokens are there, but big is ahead: 5.25
        ("try", 2.0, "probe", 1),  # 16 tokens are there: still False
        ("acquire", 3.0, "medium", 16),  # 7.25
        ("refund", 4.0, 8),  # balance 40: not enough for big, and small must not take it
        ("acquire", 5.0, "last", 1),  # 7.375 (after medium)
    ]
    run = simulate(ctx, ctx.solution.TokenBucket, 8, 64, script)
    check_same(
        run,
        grants={"big": 5.0, "small": 5.25, "medium": 7.25, "last": 7.375},
        tries={"drain": True, "probe": False},
        errors={},
        label="rate 8, burst 64",
    )


@case("a refund serves waiting calls at once, and is capped at burst", public=True)
def refund_wakes_waiters(ctx):
    # Rejects refund_does_not_wake and timer_not_rearmed.
    script = [
        ("try", 0.0, "drain", 32),
        ("acquire", 0.0, "a", 16),  # would be served at 4.0
        ("acquire", 0.0, "b", 4),  # would be served at 5.0
        ("acquire", 0.0, "c", 8),  # would be served at 7.0
        ("refund", 1.0, 18),  # balance 4 + 18 = 22: a and b now; c needs 6 more: 2.5
        ("refund", 10.0, 1000),  # capped: the bucket holds 32
        ("try", 10.0, "all", 32),
        ("try", 10.0, "more", 1),
        ("refund", 11.0, 0),  # a refund of nothing changes nothing
        ("try", 11.0, "four", 5),  # 4 tokens accrued since 10.0
    ]
    run = simulate(ctx, ctx.solution.TokenBucket, 4, 32, script)
    check_same(
        run,
        grants={"a": 1.0, "b": 1.0, "c": 2.5},
        tries={"drain": True, "all": True, "more": False, "four": False},
        errors={},
        label="rate 4, burst 32",
    )


@case("more tokens than burst: ValueError at once, and the queue is not disturbed")
def oversized_request(ctx):
    # Rejects oversized_waits_forever.
    script = [
        ("try", 0.0, "drain", 20),
        ("acquire", 0.0, "first", 10),  # 5.0
        ("acquire", 1.0, "too-big", 21),
        ("try", 1.0, "too-big-try", 400),
        ("acquire", 1.0, "second", 20),  # the whole bucket is allowed: 15.0
        ("try", 30.0, "exactly-burst", 20),
        ("try", 60.0, "too-big-idle", 21),  # also when nobody waits and the bucket is full
    ]
    run = simulate(ctx, ctx.solution.TokenBucket, 2, 20, script)
    check_same(
        run,
        grants={"first": 5.0, "second": 15.0},
        tries={"drain": True, "exactly-burst": True},
        errors={"too-big": 1.0, "too-big-try": 1.0, "too-big-idle": 60.0},
        label="rate 2, burst 20",
    )


@case("ties: calls at the very instant a waiting call's tokens have accrued")
def same_instant(ctx):
    # Rejects refund_before_settling: at t=2.0 the waiter's 8 tokens have accrued, so it
    # is served first and the refund of 6 lands in an empty bucket (6, not 8 + 6 capped
    # at 10 minus 8 = 2).
    script = [
        ("try", 0.0, "drain", 10),
        ("acquire", 0.0, "w", 8),  # due at 2.0
        ("refund", 2.0, 6),
        ("try", 2.0, "six", 6),
        ("try", 2.0, "none", 1),
        ("acquire", 3.0, "x", 8),  # 4 there, due at 4.0
        ("try", 4.0, "at-grant", 1),  # x takes its 8 at this instant: nothing is left
        ("acquire", 4.0, "y", 4),  # 5.0
        ("acquire", 4.0, "z", 4),  # 6.0
        ("try", 6.0, "after", 1),
    ]
    run = simulate(ctx, ctx.solution.TokenBucket, 4, 10, script)
    check_same(
        run,
        grants={"w": 2.0, "x": 4.0, "y": 5.0, "z": 6.0},
        tries={"drain": True, "six": True, "none": False, "at-grant": False, "after": False},
        errors={},
        label="rate 4, burst 10",
    )


@case("a burst of simultaneous requests is served one after another, in call order")
def simultaneous_burst(ctx):
    # Rejects timer_not_rearmed: each grant must schedule the next.
    script = [("acquire", 2.0, f"r{i}", n) for i, n in enumerate([60, 10, 64, 1, 1, 30, 64])]
    script.append(("try", 2.0, "t", 1))
    run = check_against_oracle(ctx, 32, 64, script, "rate 32, burst 64")
    check.equal(run.grants["r0"], 2.0, label="grant time of the first request")
    # 230 tokens are asked for and 64 are there: the last request waits (230 - 64) / 32 s.
    check.equal(run.grants["r6"], 2.0 + 166 / 32, label="grant time of the last request")


def random_script(rng, *, n_ops, burst, max_gap_ticks):
    """A random script with times on a 1/64 s grid; about a tenth of the calls are refunds."""
    script, tick = [], 0
    for i in range(n_ops):
        tick += int(rng.integers(0, max_gap_ticks + 1))  # 0: the same instant as the last call
        when = tick / 64
        roll = rng.random()
        n = int(rng.integers(1, burst + 1)) if rng.random() < 0.3 else int(rng.integers(1, 12))
        if roll < 0.55:
            script.append(("acquire", when, f"a{i}", n))
        elif roll < 0.9:
            script.append(("try", when, f"t{i}", n))
        else:
            script.append(("refund", when, int(rng.integers(0, burst // 2))))
    return script


@case("random traffic at rate 64, burst 128: every grant time and every answer")
def random_traffic(ctx):
    for trial in range(4):
        script = random_script(ctx.child_rng(trial), n_ops=120, burst=128, max_gap_ticks=24)
        check_against_oracle(ctx, 64, 128, script, f"random script {trial}")


@case("random traffic with a slow refill (0.5 tokens/s) and a fast one (1024 tokens/s)")
def random_rates(ctx):
    script = random_script(ctx.child_rng(0), n_ops=80, burst=16, max_gap_ticks=640)
    check_against_oracle(ctx, 0.5, 16, script, "rate 0.5, burst 16")
    script = random_script(ctx.child_rng(1), n_ops=150, burst=4096, max_gap_ticks=8)
    check_against_oracle(ctx, 1024, 4096, script, "rate 1024, burst 4096")
