"""Tests for capacity-planner.

Worked examples are compared with literal values and everything else with the oracle.
The property cases state what a plan means without the oracle, in exact rational
arithmetic: the chosen batch meets the objective and the next one does not; the
replica count carries the load inside the headroom and one fewer does not.
"""

from fractions import Fraction
from itertools import pairwise

from learnai.judge import case, check

COST = {"base_us": 8000, "prefill_us_per_token": 100, "decode_us_per_seq": 1000, "max_seqs": 32}
WORKLOAD = {"requests_per_s": 40, "prompt_tokens": 500, "output_tokens": 200}
PLAN_FLOATS = ("tpot_us", "requests_per_s", "tokens_per_s", "utilization")
# The statement's measured curve: (requests per second on one replica, p99 TTFT in ms).
CURVE = [(1, 180), (2, 210), (3, 290), (4, 520), (5, 1400)]


def show_plan_inputs(cost, workload, slo, headroom):
    return (
        f"cost = {cost!r}\nworkload = {workload!r}\n"
        f"tpot_slo_us = {slo}\nheadroom_percent = {headroom}\n"
    )


def call_plan(ctx, cost, workload, slo, headroom, label):
    got = ctx.call("plan", dict(cost), dict(workload), slo, headroom)
    inputs = show_plan_inputs(cost, workload, slo, headroom)
    if got is not None:
        check.that(isinstance(got, dict), f"{label}: the plan must be a dict or None", inputs)
        check.that(
            set(got) == {"batch", "replicas", *PLAN_FLOATS},
            f"{label}: the plan does not have the specified keys",
            inputs + f"got: {got!r}\n",
        )
        check.that(
            type(got["batch"]) is int and type(got["replicas"]) is int,
            f"{label}: 'batch' and 'replicas' must be Python ints",
            inputs + f"got: {got!r}\n",
        )
    return got


def check_plan(ctx, cost, workload, slo, headroom, label, expected="oracle"):
    got = call_plan(ctx, cost, workload, slo, headroom, label)
    if expected == "oracle":
        expected = ctx.expected("plan", dict(cost), dict(workload), slo, headroom)
    detail = (
        show_plan_inputs(cost, workload, slo, headroom)
        + f"expected: {expected!r}\n     got: {got!r}\n"
    )
    if expected is None or got is None:
        check.that(
            got is None and expected is None,
            f"{label}: wrong decision on whether the objective can be met at all",
            detail,
        )
        return
    for key in ("batch", "replicas"):
        check.that(
            got[key] == expected[key], f"{label}: {key!r} is {got[key]}, which is wrong", detail
        )
    for key in PLAN_FLOATS:
        check.that(
            isinstance(got[key], float)
            and abs(got[key] - expected[key]) <= 1e-9 * max(1.0, abs(expected[key])),
            f"{label}: {key!r} is wrong",
            detail,
        )


def check_curve(ctx, curve, rate, slo, headroom, label, expected="oracle"):
    got = ctx.call("plan_from_curve", list(curve), rate, slo, headroom)
    if expected == "oracle":
        expected = ctx.expected("plan_from_curve", list(curve), rate, slo, headroom)
    detail = (
        f"curve = {curve!r}\nrequests_per_s = {rate}\nslo_ms = {slo}\n"
        f"headroom_percent = {headroom}\nexpected: {expected!r}\n     got: {got!r}\n"
    )
    if expected is None or got is None:
        check.that(
            got is None and expected is None,
            f"{label}: wrong decision on whether the objective can be met at all",
            detail,
        )
        return
    check.that(
        isinstance(got, dict) and set(got) == {"capacity", "replicas"},
        f"{label}: the result must be a dict with the keys 'capacity' and 'replicas'",
        detail,
    )
    check.that(
        isinstance(got["capacity"], float)
        and abs(got["capacity"] - expected["capacity"]) <= 1e-9 * max(1.0, expected["capacity"]),
        f"{label}: 'capacity' is wrong",
        detail,
    )
    check.that(
        type(got["replicas"]) is int and got["replicas"] == expected["replicas"],
        f"{label}: 'replicas' is wrong",
        detail,
    )


@case("the worked example: 40 requests per second at a 40 ms TPOT with 25 percent headroom",
      public=True)  # fmt: skip
def worked_example(ctx):
    # Rejects prefill_ignored (batch 32), throughput_only (batch 32),
    # headroom_multiplied (16 replicas), first_token_decoded, and max_seqs_ignored
    # (third call).
    check_plan(
        ctx, COST, WORKLOAD, 40_000, 25, "plan(cost, workload, 40000, 25)",
        {"batch": 25, "tpot_us": 39125.0, "requests_per_s": 3.194888178913738,
         "tokens_per_s": 638.9776357827476, "replicas": 17, "utilization": 0.7364705882352941},
    )  # fmt: skip
    check_plan(
        ctx, COST, WORKLOAD, 40_000, 0, "plan(cost, workload, 40000, 0)",
        {"batch": 25, "tpot_us": 39125.0, "requests_per_s": 3.194888178913738,
         "tokens_per_s": 638.9776357827476, "replicas": 13, "utilization": 0.963076923076923},
    )  # fmt: skip
    check_plan(
        ctx, COST, WORKLOAD, 100_000, 25, "plan(cost, workload, 100000, 25)",
        {"batch": 32, "tpot_us": 47840.0, "requests_per_s": 3.3444816053511706,
         "tokens_per_s": 668.8963210702341, "replicas": 16, "utilization": 0.7475},
    )  # fmt: skip
    check_plan(ctx, COST, WORKLOAD, 9_000, 25, "plan(cost, workload, 9000, 25)", None)


@case("the curve example: capacity is where the measured curve crosses 400 ms", public=True)
def curve_example(ctx):
    # Rejects no_interpolation (capacity 3) and extrapolates (the 2000 ms objective).
    check_curve(
        ctx, CURVE, 40, 400, 25, "plan_from_curve(curve, 40, 400, 25)",
        {"capacity": 3.4782608695652173, "replicas": 16},
    )  # fmt: skip
    check_curve(
        ctx, CURVE, 40, 400, 0, "plan_from_curve(curve, 40, 400, 0)",
        {"capacity": 3.4782608695652173, "replicas": 12},
    )  # fmt: skip
    check_curve(
        ctx, CURVE, 40, 2000, 25, "plan_from_curve(curve, 40, 2000, 25)",
        {"capacity": 5.0, "replicas": 11},
    )  # fmt: skip
    check_curve(ctx, CURVE, 40, 150, 25, "plan_from_curve(curve, 40, 150, 25)", None)


@case("objectives from too tight to loose: no batch, a batch of one, the cap at max_seqs",
      public=True)  # fmt: skip
def objective_sweep(ctx):
    # Rejects throughput_only, max_seqs_ignored, and slo_strict (9245 us is exactly
    # the TPOT of one sequence, 40370 us that of 26, 47840 us that of 32).
    slos = (8_000, 9_244, 9_245, 9_246, 12_000, 20_000, 40_369, 40_370, 47_839, 47_840, 500_000)
    for slo in slos:
        check_plan(ctx, COST, WORKLOAD, slo, 20, f"plan at a TPOT objective of {slo} us")


@case("loads from one request per second up: the replica count steps at exact multiples",
      public=True)  # fmt: skip
def load_sweep(ctx):
    # Rejects ceiling_omitted and headroom_multiplied.
    for rate in (1, 2, 3, 4, 7, 10, 39, 40, 41, 100, 1000, 12_345):
        for headroom in (0, 10, 50):
            workload = {**WORKLOAD, "requests_per_s": rate}
            check_plan(
                ctx, COST, workload, 40_000, headroom,
                f"plan for {rate} requests per second with {headroom} percent headroom",
            )  # fmt: skip


@case("the load is an exact multiple of the usable capacity: no extra replica")
def exact_multiples(ctx):
    # Rejects float_ceiling: each of these quotients is a whole number, and the float
    # division lands just above it.
    for cost, workload, slo, headroom in (
        ({"base_us": 2000, "prefill_us_per_token": 100, "decode_us_per_seq": 1000, "max_seqs": 16},
         {"requests_per_s": 125, "prompt_tokens": 200, "output_tokens": 200}, 20_000, 50),
        ({"base_us": 16000, "prefill_us_per_token": 100, "decode_us_per_seq": 2000, "max_seqs": 8},
         {"requests_per_s": 100, "prompt_tokens": 100, "output_tokens": 400}, 50_000, 20),
        ({"base_us": 8000, "prefill_us_per_token": 100, "decode_us_per_seq": 2000, "max_seqs": 8},
         {"requests_per_s": 200, "prompt_tokens": 500, "output_tokens": 50}, 50_000, 10),
        ({"base_us": 4000, "prefill_us_per_token": 100, "decode_us_per_seq": 2000, "max_seqs": 16},
         {"requests_per_s": 125, "prompt_tokens": 400, "output_tokens": 200}, 100_000, 0),
    ):  # fmt: skip
        label = f"plan for {workload['requests_per_s']} requests per second, an exact fit"
        check_plan(ctx, cost, workload, slo, headroom, label)
    # A capacity of exactly 2.5 requests per second per replica, read off a curve.
    curve = [(2, 100), (3, 300)]
    for rate, headroom in ((10, 0), (10, 20), (11, 0), (2, 20), (1, 0), (25, 50)):
        check_curve(
            ctx, curve, rate, 200, headroom,
            f"plan_from_curve, capacity 2.5, {rate} requests per second, headroom {headroom}",
        )  # fmt: skip


@case("prompts from one token to very long, outputs from one token up")
def length_extremes(ctx):
    # Rejects prefill_ignored and first_token_decoded (one output token: no decode step).
    for prompt, output in (
        (1, 1),
        (4000, 1),
        (1, 2),
        (8000, 2),
        (16, 1000),
        (32_000, 64),
        (500, 2000),
    ):
        workload = {"requests_per_s": 25, "prompt_tokens": prompt, "output_tokens": output}
        for slo in (10_000, 60_000, 1_000_000):
            check_plan(
                ctx, COST, workload, slo, 30,
                f"plan for prompts of {prompt} and outputs of {output} tokens, objective {slo} us",
            )  # fmt: skip


@case("curves: flat, a single point, the objective on a measured point, a dip after the knee")
def curve_shapes(ctx):
    # Rejects last_crossing (the dip), no_interpolation, and extrapolates.
    dip = [(1, 100), (2, 150), (3, 450), (4, 380), (5, 390), (6, 900)]
    for slo in (99, 100, 150, 151, 300, 385, 400, 449, 450, 899, 900, 5000):
        check_curve(ctx, dip, 30, slo, 20, f"a curve with a dip after the knee, objective {slo} ms")
    flat = [(2, 200), (4, 200), (8, 200), (16, 200)]
    for slo in (199, 200, 201):
        check_curve(ctx, flat, 100, slo, 10, f"a flat curve, objective {slo} ms")
    for slo in (249, 250, 1000):
        check_curve(ctx, [(7, 250)], 50, slo, 0, f"a single measured point, objective {slo} ms")
    rising = [(10, 50), (20, 60), (30, 70)]
    check_curve(ctx, rising, 500, 400, 25, "every point passes and the curve is still rising")
    steep = [(1, 10), (1000, 20), (1001, 100_000)]
    for slo in (20, 21, 50_000):
        check_curve(
            ctx, steep, 123_456, slo, 33, f"a cliff after 1000 requests per second, {slo} ms"
        )


def random_plan_inputs(rng):
    cost = {
        "base_us": int(rng.integers(1, 30_000)),
        "prefill_us_per_token": int(rng.integers(1, 400)),
        "decode_us_per_seq": int(rng.integers(1, 4000)),
        "max_seqs": int(rng.integers(1, 129)),
    }
    workload = {
        "requests_per_s": int(rng.integers(1, 2000)),
        "prompt_tokens": int(rng.integers(1, 8000)),
        "output_tokens": int(rng.integers(1, 1000)),
    }
    slo = int(rng.integers(1, 200_000))
    headroom = int(rng.integers(0, 91))
    return cost, workload, slo, headroom


def tpot(cost, workload, batch):
    """The model's mean time per output token at `batch` sequences, exactly."""
    p, o = workload["prompt_tokens"], workload["output_tokens"]
    per_slot = cost["prefill_us_per_token"] * p + cost["decode_us_per_seq"] * (o - 1)
    return Fraction(o * cost["base_us"] + batch * per_slot, o)


@case("property: the batch meets the objective and the next one does not (or is the cap)")
def batch_is_maximal(ctx):
    def prop(rng):
        cost, workload, slo, headroom = random_plan_inputs(rng)
        got = call_plan(ctx, cost, workload, slo, headroom, "plan")
        inputs = show_plan_inputs(cost, workload, slo, headroom) + f"got: {got!r}\n"
        if got is None:
            check.that(
                tpot(cost, workload, 1) > slo,
                "plan returned None although a single sequence meets the objective",
                inputs,
            )
            return
        b = got["batch"]
        check.that(1 <= b <= cost["max_seqs"], "plan: 'batch' is outside 1..max_seqs", inputs)
        check.that(
            tpot(cost, workload, b) <= slo,
            "plan: the TPOT of the chosen batch exceeds the objective",
            inputs,
        )
        check.that(
            b == cost["max_seqs"] or tpot(cost, workload, b + 1) > slo,
            "plan: one more sequence would still meet the objective",
            inputs,
        )

    check.forall(prop, trials=200)


@case("property: the replicas carry the load inside the headroom, and one fewer does not")
def replicas_are_minimal(ctx):
    def within(rate, replicas, capacity, headroom):
        return rate * 100 <= replicas * capacity * (100 - headroom)

    def prop(rng):
        cost, workload, slo, headroom = random_plan_inputs(rng)
        got = call_plan(ctx, cost, workload, slo, headroom, "plan")
        if got is not None and 1 <= got["batch"] <= cost["max_seqs"]:
            inputs = show_plan_inputs(cost, workload, slo, headroom) + f"got: {got!r}\n"
            # Judged at the batch the plan itself chose, so that this property is
            # about the replica count alone.
            capacity = (
                got["batch"]
                * 10**6
                / (tpot(cost, workload, got["batch"]) * workload["output_tokens"])
            )
            rate, r = workload["requests_per_s"], got["replicas"]
            check.that(r >= 1, "plan: 'replicas' must be at least 1", inputs)
            check.that(
                within(rate, r, capacity, headroom),
                "plan: the replicas would run above the allowed share of their capacity",
                inputs,
            )
            check.that(
                r == 1 or not within(rate, r - 1, capacity, headroom),
                "plan: one replica fewer would still carry the load inside the headroom",
                inputs,
            )

        points = int(rng.integers(1, 8))
        rates = sorted(int(v) for v in rng.choice(range(1, 400), size=points, replace=False))
        latency, curve = int(rng.integers(20, 200)), []
        for rate in rates:
            latency = max(1, latency + int(rng.integers(-30, 200)))
            curve.append((rate, latency))
        slo = max(curve[0][1], int(rng.integers(20, 1500)))
        demand, headroom = int(rng.integers(1, 5000)), int(rng.integers(0, 91))
        got = ctx.call("plan_from_curve", list(curve), demand, slo, headroom)
        inputs = (
            f"curve = {curve!r}\nrequests_per_s = {demand}\nslo_ms = {slo}\n"
            f"headroom_percent = {headroom}\ngot: {got!r}\n"
        )
        check.that(
            isinstance(got, dict) and type(got.get("replicas")) is int and got["replicas"] >= 1,
            "plan_from_curve: expected a dict with an int 'replicas' >= 1 for a feasible curve",
            inputs,
        )
        capacity = Fraction(curve[-1][0])
        for (r0, l0), (r1, l1) in pairwise(curve):
            if l1 > slo:
                capacity = r0 + Fraction((slo - l0) * (r1 - r0), l1 - l0)
                break
        r = got["replicas"]
        check.that(
            within(demand, r, capacity, headroom),
            "plan_from_curve: the replicas would run above the allowed share of their capacity",
            inputs,
        )
        check.that(
            r == 1 or not within(demand, r - 1, capacity, headroom),
            "plan_from_curve: one replica fewer would still carry the load inside the headroom",
            inputs,
        )

    check.forall(prop, trials=200)


@case("random cost models, workloads, objectives, and curves against the oracle")
def random_plans(ctx):
    rng = ctx.rng
    for i in range(150):
        cost, workload, slo, headroom = random_plan_inputs(rng)
        check_plan(ctx, cost, workload, slo, headroom, f"plan(random case {i})")
    for i in range(150):
        points = int(rng.integers(1, 10))
        rates = sorted(int(v) for v in rng.choice(range(1, 300), size=points, replace=False))
        latency, curve = int(rng.integers(20, 300)), []
        for rate in rates:
            latency = max(1, latency + int(rng.integers(-40, 250)))
            curve.append((rate, latency))
        slo = int(rng.integers(10, 1500))
        check_curve(
            ctx, curve, int(rng.integers(1, 3000)), slo, int(rng.integers(0, 91)),
            f"plan_from_curve(random case {i})",
        )  # fmt: skip
    kept_cost, kept_workload, kept_curve = dict(COST), dict(WORKLOAD), list(CURVE)
    ctx.solution.plan(kept_cost, kept_workload, 40_000, 25)
    ctx.solution.plan_from_curve(kept_curve, 40, 400, 25)
    check.that(
        kept_cost == COST and kept_workload == WORKLOAD and kept_curve == CURVE,
        "the functions must not modify their arguments",
    )
