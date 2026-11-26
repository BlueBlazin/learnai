"""Tests for latency-metrics.

The worked example is compared with literal values. Everything else is compared with
`expected_metrics` below, which restates the definitions of the statement, and the
random sweep is also compared with the oracle.
"""

from learnai.judge import case, check

KEYS = ("ttft", "itl", "tpot", "e2e", "tokens_per_s")
TOL = {"rtol": 1e-9, "atol": 1e-9}

# The worked example. Every value is a binary fraction, so the results are exact.
T_START = 12.0
TOKEN_TIMES = [12.5, 12.5625, 12.625, 12.75, 13.0]


def expected_metrics(t_start, times):
    n = len(times)
    e2e = times[-1] - t_start
    return {
        "ttft": times[0] - t_start,
        "itl": [times[i] - times[i - 1] for i in range(1, n)],
        "tpot": (times[-1] - times[0]) / (n - 1) if n > 1 else None,
        "e2e": e2e,
        "tokens_per_s": n / e2e if e2e > 0 else None,
    }


def describe(t_start, times):
    shown = ", ".join(repr(t) for t in times[:12]) + (", ..." if len(times) > 12 else "")
    return f"t_start = {t_start!r}\ntoken_times ({len(times)} values) = [{shown}]\n"


def check_metrics(ctx, t_start, times, label, expected=None):
    if expected is None:
        expected = expected_metrics(t_start, times)
    got = ctx.call("request_metrics", t_start, list(times))
    inputs = describe(t_start, times)
    check.that(
        isinstance(got, dict) and set(got) == set(KEYS),
        f"{label}: the result must be a dict with exactly the keys {', '.join(KEYS)}",
        inputs + f"got: {got!r}\n",
    )
    for key in KEYS:
        want, value = expected[key], got[key]
        detail = inputs + f"expected {key}: {want!r}\n     got {key}: {value!r}\n"
        name = f"{label}: {key}"
        if want is None:
            check.that(value is None, f"{name} must be None for this input", detail)
        elif key == "itl":
            check.that(
                isinstance(value, list) and all(type(v) is float for v in value),
                f"{name} must be a list of floats",
                detail,
            )
            check.that(
                len(value) == len(want),
                f"{name} has {len(value)} entries, expected {len(want)} (one per gap between "
                "tokens)",
                detail,
            )
            check.allclose(value, want, label=name, **TOL)
        else:
            check.that(type(value) is float, f"{name} must be a float", detail)
            check.allclose(value, want, label=name, **TOL)
    return got


@case("the worked example: five tokens, the first after 0.5 s", public=True)
def worked_example(ctx):
    # Rejects tpot_includes_first (0.2), tpot_divides_by_n (0.1), itl_from_request_start,
    # tps_generation_only (10.0), and ttft_ignores_start (12.5).
    check_metrics(
        ctx,
        T_START,
        TOKEN_TIMES,
        "request_metrics(12.0, [12.5, 12.5625, 12.625, 12.75, 13.0])",
        {
            "ttft": 0.5,
            "itl": [0.0625, 0.0625, 0.125, 0.25],
            "tpot": 0.125,
            "e2e": 1.0,
            "tokens_per_s": 5.0,
        },
    )


@case("a single token: no gaps, so itl is empty and tpot is None", public=True)
def single_token(ctx):
    # Rejects single_token_tpot_zero.
    check_metrics(ctx, 3.0, [3.75], "request_metrics(3.0, [3.75])")
    check_metrics(ctx, 0.0, [2.0], "request_metrics(0.0, [2.0])")


@case("two tokens, and a clock that starts at 0", public=True)
def two_tokens(ctx):
    check_metrics(ctx, 0.0, [0.25, 0.75], "request_metrics(0.0, [0.25, 0.75])")
    check_metrics(ctx, 40.0, [41.0, 41.5], "request_metrics(40.0, [41.0, 41.5])")


@case("a stall in the middle of the stream stays at its position in itl", public=True)
def stall_in_the_middle(ctx):
    # Rejects itl_sorted: the long gap is the third of six, not the last.
    times = [5.25, 5.3125, 5.375, 7.375, 7.4375, 7.5, 7.5625]
    check_metrics(ctx, 5.0, times, "a stream that stalls for 2 s after its third token")


@case("tokens that arrive together: several tokens in one chunk give gaps of 0")
def simultaneous_tokens(ctx):
    times = [1.5, 1.5, 1.5, 2.0, 2.0, 2.25]
    check_metrics(ctx, 1.0, times, "tokens delivered in chunks of 3, 2, and 1")
    check_metrics(ctx, 1.0, [1.5, 1.5], "two tokens in a single chunk")


@case("zero end-to-end latency: tokens_per_s is None")
def zero_duration(ctx):
    # Rejects zero_e2e_infinite (and a division by zero).
    check_metrics(ctx, 7.0, [7.0], "one token at the request's own timestamp")
    check_metrics(ctx, 7.0, [7.0, 7.0, 7.0], "three tokens at the request's own timestamp")
    check_metrics(ctx, 7.0, [7.0, 7.5], "a first token at the request's own timestamp")


@case("an empty token list raises ValueError")
def empty_raises(ctx):
    # Rejects empty_is_not_an_error.
    check.raises(
        ValueError, ctx.solution.request_metrics, 1.0, [], label="request_metrics(1.0, [])"
    )


@case("the caller's list is not modified")
def input_unchanged(ctx):
    times = list(TOKEN_TIMES)
    ctx.solution.request_metrics(T_START, times)
    check.that(times == TOKEN_TIMES, "request_metrics modified token_times")


@case("timestamps from a wall clock: Unix times around 1.8e9 with millisecond gaps")
def wall_clock_times(ctx):
    # Rejects ttft_ignores_start by a wide margin; differences of large timestamps
    # are still exact, so the tolerance is not strained.
    t_start = 1_794_000_000.0 + float(ctx.rng.integers(0, 10_000))
    gaps = ctx.rng.integers(1, 40, size=200) / 1024.0
    times = (t_start + 0.375 + gaps.cumsum()).tolist()
    check_metrics(ctx, t_start, times, "200 tokens with Unix timestamps")


@case("random streams of 1 to 400 tokens, compared with the definitions and the oracle")
def random_streams(ctx):
    for n in (1, 2, 3, 5, 17, 64, 400):
        t_start = float(ctx.rng.uniform(0.0, 5000.0))
        ttft = float(ctx.rng.uniform(0.02, 3.0))
        gaps = ctx.rng.exponential(0.03, size=n - 1)
        # Some tokens share a chunk, and one gap is a stall.
        gaps[ctx.rng.random(n - 1) < 0.2] = 0.0
        if n > 3:
            gaps[int(ctx.rng.integers(0, n - 1))] = 1.5
        times = [t_start + ttft]
        for gap in gaps:
            times.append(times[-1] + float(gap))
        got = check_metrics(ctx, t_start, times, f"a random stream of {n} tokens")
        want = ctx.expected("request_metrics", t_start, list(times))
        for key in KEYS:
            if want[key] is None:
                check.that(got[key] is None, f"a random stream of {n} tokens: {key} must be None")
            else:
                check.allclose(got[key], want[key], label=f"{key} of {n} tokens", **TOL)
