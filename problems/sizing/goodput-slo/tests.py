"""Tests for goodput-slo.

The worked examples are compared with literal values, everything else with the oracle.
Attainments are compared with a tolerance of 1e-12; rates returned by `goodput` are
keys of `runs` and are compared exactly.
"""

from fractions import Fraction

from learnai.judge import case, check

SLO = {"ttft": 0.5, "tpot": 0.05}


def req(ttft, tpot, e2e, ok=True):
    return {"ok": ok, "ttft": ttft, "tpot": tpot, "e2e": e2e}


FAILED = {"ok": False, "ttft": None, "tpot": None, "e2e": None}

# The statement's ten requests, measured over four seconds.
TEN = [
    req(0.20, 0.030, 1.7),
    req(0.25, 0.040, 2.2),
    req(0.90, 0.030, 2.4),  # slow first token
    req(0.30, 0.050, 2.8),  # tpot exactly at the objective
    req(0.35, 0.080, 4.3),  # slow decode
    req(0.50, 0.045, 2.7),  # ttft exactly at the objective
    req(0.40, None, 0.4),  # one token
    FAILED,
    req(0.45, 0.049, 2.9),
    req(0.60, 0.070, 4.1),  # misses both
]


def copy_requests(requests):
    return [dict(r) for r in requests]


def check_attainment(ctx, requests, slo, label, expected=None):
    got = ctx.call("slo_attainment", copy_requests(requests), dict(slo))
    check.that(isinstance(got, float), f"{label}: the result must be a float", f"got: {got!r}")
    if expected is None:
        expected = ctx.expected("slo_attainment", copy_requests(requests), dict(slo))
    check.allclose(got, expected, rtol=1e-12, atol=1e-12, label=label)


def check_request_goodput(ctx, requests, slo, duration, label, expected=None):
    got = ctx.call("request_goodput", copy_requests(requests), dict(slo), duration)
    check.that(isinstance(got, float), f"{label}: the result must be a float", f"got: {got!r}")
    if expected is None:
        expected = ctx.expected("request_goodput", copy_requests(requests), dict(slo), duration)
    check.allclose(got, expected, rtol=1e-12, atol=1e-12, label=label)


def good_count(requests, slo):
    """Counted independently of the oracle, for the boundary promise below."""
    count = 0
    for r in requests:
        if r["ok"] and all(r[k] is None or r[k] <= limit for k, limit in slo.items()):
            count += 1
    return count


def check_goodput(ctx, runs, slo, target, label, expected="oracle"):
    # The statement promises that no run sits where good / n >= target and the exact
    # comparison disagree.
    for requests in runs.values():
        good, n = good_count(requests, slo), len(requests)
        assert (good / n >= target) == (Fraction(good, n) >= Fraction(str(target)))
    fresh = {rate: copy_requests(requests) for rate, requests in runs.items()}
    got = ctx.call("goodput", fresh, dict(slo), target)
    if expected == "oracle":
        fresh = {rate: copy_requests(requests) for rate, requests in runs.items()}
        expected = ctx.expected("goodput", fresh, dict(slo), target)
    detail = "".join(
        f"rate {rate}: {good_count(requests, slo)} of {len(requests)} requests good\n"
        for rate, requests in runs.items()
    )
    check.that(
        got == expected and type(got) is type(expected),
        f"{label}: got {got!r}, expected {expected!r}",
        f"runs, in dict order:\n{detail}target = {target}\nexpected: {expected!r}\n     got: {got!r}",
    )


def run_with(n, good, slo=SLO):
    """n requests of which `good` meet `slo`; the others miss the ttft objective."""
    fast = req(slo.get("ttft", 1.0) * 0.5, 0.02, 1.0)
    slow = req(slo.get("ttft", 1.0) * 3.0, 0.02, 2.0)
    return [dict(fast) for _ in range(good)] + [dict(slow) for _ in range(n - good)]


@case("the worked example: ten requests, six of them good", public=True)
def worked_example(ctx):
    # Rejects mean_latency (the means pass), either_slo, strict_less, failures_dropped,
    # single_token_violates, errors_are_fast, and raw_throughput.
    check_attainment(ctx, TEN, SLO, "slo_attainment(requests, slo)", expected=0.6)
    check_request_goodput(ctx, TEN, SLO, 4.0, "request_goodput(requests, slo, 4.0)", expected=1.5)
    check_attainment(
        ctx, TEN, {**SLO, "e2e": 2.5}, "slo_attainment with an e2e objective of 2.5 s", expected=0.3
    )
    check_attainment(ctx, TEN, {}, "slo_attainment with no objectives", expected=0.9)


@case("the rate sweep example: a run above the knee passes by luck", public=True)
def rate_sweep(ctx):
    # Rejects highest_passing_rate (8.0 passes after 4.0 failed) and dict_order.
    runs = {
        8.0: run_with(40, 36),
        1.0: run_with(20, 20),
        4.0: run_with(40, 34),
        2.0: run_with(20, 19),
        16.0: run_with(40, 11),
    }
    check_goodput(ctx, runs, SLO, 0.9, "goodput(runs, slo, 0.9)", expected=2.0)
    check_goodput(ctx, runs, SLO, 0.8, "goodput(runs, slo, 0.8)", expected=8.0)
    check_goodput(ctx, runs, SLO, 0.99, "goodput(runs, slo, 0.99)", expected=1.0)
    check_goodput(ctx, runs, SLO, 1.0, "goodput(runs, slo, 1.0)", expected=1.0)


@case("each objective alone, pairs, all three, and none", public=True)
def objective_subsets(ctx):
    # Rejects either_slo.
    for slo in (
        {"ttft": 0.5},
        {"tpot": 0.05},
        {"e2e": 2.75},
        {"ttft": 0.5, "e2e": 2.75},
        {"tpot": 0.05, "e2e": 2.75},
        {"ttft": 0.5, "tpot": 0.05, "e2e": 2.75},
        {},
    ):
        check_attainment(ctx, TEN, slo, f"slo_attainment with objectives {slo!r}")
        check_request_goodput(ctx, TEN, slo, 4.0, f"request_goodput with objectives {slo!r}")


@case("failed requests are never good and stay in the denominator", public=True)
def failed_requests(ctx):
    # Rejects failures_dropped and errors_are_fast. A rejection that came back at once
    # carries latencies well inside every objective.
    rejected = {"ok": False, "ttft": 0.001, "tpot": None, "e2e": 0.001}
    fast = req(0.1, 0.01, 0.5)
    for requests, label in (
        ([fast, FAILED, FAILED, FAILED], "one success and three failures"),
        ([rejected, rejected, fast, fast], "two instant rejections and two successes"),
        ([FAILED, rejected], "nothing but failures"),
        ([fast] * 5, "no failures"),
    ):
        check_attainment(ctx, requests, SLO, f"slo_attainment, {label}")
        check_attainment(ctx, requests, {}, f"slo_attainment with no objectives, {label}")
        check_request_goodput(ctx, requests, SLO, 2.0, f"request_goodput, {label}")


@case("latencies exactly at an objective meet it; one-token responses meet the tpot objective")
def boundaries(ctx):
    # Rejects strict_less and single_token_violates.
    slo = {"ttft": 0.25, "tpot": 0.0625, "e2e": 2.0}
    requests = [
        req(0.25, 0.0625, 2.0),
        req(0.25, 0.03, 1.0),
        req(0.1, 0.0625, 1.0),
        req(0.1, 0.03, 2.0),
        req(0.2500001, 0.03, 1.0),
        req(0.1, 0.0625001, 1.0),
        req(0.1, 0.03, 2.0000001),
        req(0.1, None, 0.1),
        req(0.3, None, 0.3),
        req(0.2, None, 2.5),
    ]
    check_attainment(ctx, requests, slo, "slo_attainment at the boundaries")
    check_attainment(ctx, requests[:4], slo, "slo_attainment, four requests exactly at objectives")
    check_attainment(ctx, requests[7:], slo, "slo_attainment, one-token responses")
    check_request_goodput(ctx, requests, slo, 0.5, "request_goodput at the boundaries")


@case("a slow minority behind a fast mean, and one outlier among fast requests")
def mean_is_not_the_slo(ctx):
    # Rejects mean_latency in both directions.
    slo = {"ttft": 1.0, "tpot": 0.05}
    mostly_fast = [req(0.1, 0.02, 1.0)] * 70 + [req(1.5, 0.02, 2.5)] * 30  # mean ttft 0.52
    one_outlier = [req(0.2, 0.02, 1.0)] * 99 + [req(120.0, 0.02, 121.0)]  # mean ttft 1.398
    check_attainment(ctx, mostly_fast, slo, "slo_attainment, 30 percent slow first tokens")
    check_attainment(ctx, one_outlier, slo, "slo_attainment, one request of 100 very slow")
    runs = {2.0: one_outlier, 4.0: mostly_fast}
    check_goodput(ctx, runs, slo, 0.95, "goodput over those two runs")


@case("request_goodput: empty list, long and short durations, throughput for comparison")
def goodput_rate(ctx):
    # Rejects raw_throughput.
    check_request_goodput(ctx, [], SLO, 10.0, "request_goodput of no requests")
    requests = run_with(120, 45) + [FAILED] * 7
    for duration in (0.5, 1.0, 30.0, 61.25):
        check_request_goodput(ctx, requests, SLO, duration, f"request_goodput over {duration} s")


@case("goodput: attainment exactly at the target, all rates pass, the lowest rate fails")
def sweep_edges(ctx):
    # Rejects target_strict and highest_passing_rate.
    exact = {1.0: run_with(10, 10), 2.0: run_with(10, 9), 3.0: run_with(20, 17)}
    check_goodput(ctx, exact, SLO, 0.9, "goodput, attainment exactly 0.9 at rate 2.0")
    all_pass = {0.5: run_with(8, 8), 1.5: run_with(8, 8), 2.5: run_with(8, 7)}
    check_goodput(ctx, all_pass, SLO, 0.75, "goodput when every rate passes")
    lowest_fails = {1.0: run_with(10, 5), 2.0: run_with(10, 10), 4.0: run_with(10, 10)}
    check_goodput(ctx, lowest_fails, SLO, 0.9, "goodput when the lowest rate fails")
    single = {3.5: run_with(4, 4)}
    check_goodput(ctx, single, SLO, 1.0, "goodput with a single passing run")
    check_goodput(ctx, {3.5: run_with(4, 3)}, SLO, 1.0, "goodput with a single failing run")
    ints = {4: run_with(10, 10), 2: run_with(10, 10), 8: run_with(10, 2)}
    check_goodput(ctx, ints, SLO, 0.5, "goodput with integer rates in descending order")


@case("empty input raises ValueError, and the arguments are not modified")
def errors_and_purity(ctx):
    check.raises(ValueError, ctx.solution.slo_attainment, [], dict(SLO), label="slo_attainment([])")
    requests, slo = copy_requests(TEN), dict(SLO)
    runs = {2.0: copy_requests(TEN), 1.0: copy_requests(TEN)}
    ctx.solution.slo_attainment(requests, slo)
    ctx.solution.request_goodput(requests, slo, 4.0)
    ctx.solution.goodput(runs, slo, 0.5)
    check.that(requests == TEN and slo == SLO, "the functions must not modify requests or slo")
    check.that(
        list(runs) == [2.0, 1.0] and runs[2.0] == TEN and runs[1.0] == TEN,
        "goodput must not modify or reorder runs",
    )


def random_request(rng):
    if rng.random() < 0.08:
        return dict(FAILED)
    ttft = round(float(rng.lognormal(-1.2, 0.7)), 3)
    tokens = int(rng.integers(1, 200))
    tpot = None if tokens == 1 else round(float(rng.lognormal(-3.3, 0.5)), 4)
    e2e = ttft + (0.0 if tpot is None else tpot * (tokens - 1))
    return req(ttft, tpot, round(e2e, 3))


@case("random sweeps: lognormal latencies, failures, one-token responses, shuffled rates")
def random_sweeps(ctx):
    rng = ctx.rng
    for i in range(25):
        slo = {}
        if rng.random() < 0.8:
            slo["ttft"] = round(float(rng.uniform(0.2, 1.0)), 2)
        if rng.random() < 0.8:
            slo["tpot"] = round(float(rng.uniform(0.03, 0.08)), 3)
        if rng.random() < 0.4:
            slo["e2e"] = round(float(rng.uniform(1.0, 8.0)), 1)
        rates = [float(r) for r in rng.permutation(np_range(rng))]
        runs = {}
        for rate in rates:
            n = int(rng.integers(30, 80))
            slowdown = 1.0 + 0.15 * rate * float(rng.uniform(0.5, 1.5))
            requests = []
            for _ in range(n):
                r = random_request(rng)
                if r["ok"]:
                    r["ttft"] = round(r["ttft"] * slowdown, 3)
                    r["e2e"] = round(r["e2e"] * slowdown, 3)
                requests.append(r)
            runs[rate] = requests
        first = runs[rates[0]]
        check_attainment(ctx, first, slo, f"slo_attainment(random run {i})")
        check_request_goodput(ctx, first, slo, 12.5, f"request_goodput(random run {i})")
        for target in (0.515, 0.715, 0.915):
            check_goodput(ctx, runs, slo, target, f"goodput(random sweep {i}, target {target})")


def np_range(rng):
    """Between three and seven distinct rates from 1 to 12 requests per second."""
    count = int(rng.integers(3, 8))
    return sorted(rng.choice(range(1, 13), size=count, replace=False).tolist())
